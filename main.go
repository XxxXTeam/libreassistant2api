package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

/*
 * tokenRecord 记录单次请求的 token 用量和时间
 */
type tokenRecord struct {
	Time   time.Time
	Tokens int64
}

/*
 * ModelStats 每模型的统计信息
 * 字段说明：
 *   Requests     - 该模型的成功请求数
 *   InputTokens  - 该模型的输入 token 总量
 *   OutputTokens - 该模型的输出 token 总量
 */
type ModelStats struct {
	Requests     int64
	InputTokens  int64
	OutputTokens int64
}

type Stats struct {
	mu                sync.RWMutex
	StartTime         time.Time
	TotalRequests     int64
	SuccessRequests   int64
	FailedRequests    int64
	TotalInputTokens  int64
	TotalOutputTokens int64
	MultimodalCalls   int64
	ModelUsage        map[string]*ModelStats
	RequestTimestamps []time.Time
	TokenTimestamps   []tokenRecord
}

/* 全局统计实例 */
var stats = &Stats{
	StartTime:  time.Now(),
	ModelUsage: make(map[string]*ModelStats),
}

type Config struct {
	ApiKeys        []string `json:"api_keys"`
	Port           string   `json:"port"`
	UpstreamURL    string   `json:"upstream_url"`
	MaxBodySize    int64    `json:"max_body_size"`
	ScannerBufSize int      `json:"scanner_buf_size"`
}

var config = Config{
	Port:           "8080",
	UpstreamURL:    "https://libreassistant.vercel.app/api/ai",
	MaxBodySize:    100 * 1024 * 1024,
	ScannerBufSize: 10 * 1024 * 1024,
}

func loadConfig() {
	data, err := os.ReadFile("config.json")
	if err != nil {
		slog.Warn("配置文件不存在，创建默认配置", "file", "config.json")
		defaultCfg, _ := json.MarshalIndent(config, "", "  ")
		os.WriteFile("config.json", defaultCfg, 0644)
		return
	}
	if err := json.Unmarshal(data, &config); err != nil {
		slog.Error("配置文件解析失败，使用默认配置", "error", err)
		return
	}
	if config.Port == "" {
		config.Port = "8080"
	}
	if config.UpstreamURL == "" {
		config.UpstreamURL = "https://libreassistant.vercel.app/api/ai"
	}
	if config.MaxBodySize == 0 {
		config.MaxBodySize = 100 * 1024 * 1024
	}
	if config.ScannerBufSize == 0 {
		config.ScannerBufSize = 10 * 1024 * 1024
	}
	slog.Info("配置加载成功",
		"port", config.Port,
		"upstream_url", config.UpstreamURL,
		"api_keys", len(config.ApiKeys),
		"max_body_size", config.MaxBodySize,
		"scanner_buf_size", config.ScannerBufSize,
	)
}

/*
 * httpClient 共享 HTTP 客户端
 * 通过连接池复用 TCP 连接，提高并发吞吐量
 */
var httpClient = &http.Client{
	Timeout: 300 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        200,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
		ForceAttemptHTTP2:   true,
	},
}

func (s *Stats) recordRequest() {
	atomic.AddInt64(&s.TotalRequests, 1)
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()
	/* 清理超过 5 分钟的时间戳并追加当前时间 */
	cutoff := now.Add(-5 * time.Minute)
	start := 0
	for start < len(s.RequestTimestamps) && s.RequestTimestamps[start].Before(cutoff) {
		start++
	}
	kept := s.RequestTimestamps[start:]
	newTS := make([]time.Time, len(kept)+1)
	copy(newTS, kept)
	newTS[len(kept)] = now
	s.RequestTimestamps = newTS
	/* 清理超过 5 分钟的 token 记录（拷贝到新切片释放旧内存） */
	tStart := 0
	for tStart < len(s.TokenTimestamps) && s.TokenTimestamps[tStart].Time.Before(cutoff) {
		tStart++
	}
	if tStart > 0 {
		keptTokens := make([]tokenRecord, len(s.TokenTimestamps)-tStart)
		copy(keptTokens, s.TokenTimestamps[tStart:])
		s.TokenTimestamps = keptTokens
	}
}

func (s *Stats) recordSuccess(model string, inputTokens, outputTokens int64) {
	atomic.AddInt64(&s.SuccessRequests, 1)
	atomic.AddInt64(&s.TotalInputTokens, inputTokens)
	atomic.AddInt64(&s.TotalOutputTokens, outputTokens)
	s.mu.Lock()
	/* 记录每模型维度的统计 */
	if model != "" {
		ms, ok := s.ModelUsage[model]
		if !ok {
			ms = &ModelStats{}
			s.ModelUsage[model] = ms
		}
		ms.Requests++
		ms.InputTokens += inputTokens
		ms.OutputTokens += outputTokens
	}
	/* 记录 token 时间戳用于 TPM 计算 */
	total := inputTokens + outputTokens
	if total > 0 {
		s.TokenTimestamps = append(s.TokenTimestamps, tokenRecord{
			Time:   time.Now(),
			Tokens: total,
		})
	}
	s.mu.Unlock()
}

/*
 * recordMultimodal 记录多模态请求（包含图片的请求）
 */
func (s *Stats) recordMultimodal() {
	atomic.AddInt64(&s.MultimodalCalls, 1)
}

func (s *Stats) recordFailure() {
	atomic.AddInt64(&s.FailedRequests, 1)
}

func (s *Stats) getRPM() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cutoff := time.Now().Add(-1 * time.Minute)
	count := 0
	for i := len(s.RequestTimestamps) - 1; i >= 0; i-- {
		if s.RequestTimestamps[i].Before(cutoff) {
			break
		}
		count++
	}
	return count
}

func (s *Stats) getAvgRPM() float64 {
	minutes := time.Since(s.StartTime).Minutes()
	if minutes < 1 {
		minutes = 1
	}
	return float64(atomic.LoadInt64(&s.TotalRequests)) / minutes
}
func (s *Stats) getTPM() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	cutoff := time.Now().Add(-1 * time.Minute)
	var total int64
	for i := len(s.TokenTimestamps) - 1; i >= 0; i-- {
		if s.TokenTimestamps[i].Time.Before(cutoff) {
			break
		}
		total += s.TokenTimestamps[i].Tokens
	}
	return total
}

/*
 * 字段说明：
 *   ID              - 上游模型标识（如 google/gemini-3-flash-preview）
 *   Name            - 模型显示名称
 *   Description     - 模型描述
 *   OwnedBy         - 所属提供商
 *   Reasoning       - 推理模式："always"=始终启用, "toggle"=可切换, "off"=不支持
 *   ThinkingModelID - 启用思考时切换到的模型 ID（仅特殊模型使用，如 kimi-k2）
 *   Vision          - 是否支持视觉输入
 *   ToolUse         - 是否支持工具调用
 *   ReasoningEfforts - 支持的推理强度列表（如 ["low","medium","high"]）
 *   DefaultEffort   - 默认推理强度
 */
type ModelInfo struct {
	ID               string
	Name             string
	Description      string
	OwnedBy          string
	Reasoning        string
	ThinkingModelID  string
	Vision           bool
	ToolUse          bool
	ReasoningEfforts []string
	DefaultEffort    string
}

var modelRegistry = []ModelInfo{
	{ID: "deepseek/deepseek-v3.2-speciale", Name: "DeepSeek V3.2 Speciale", Description: "High-compute SOTA variant designed for complex math & STEM tasks.", OwnedBy: "DeepSeek", Reasoning: "always", ToolUse: false},
	{ID: "deepseek/deepseek-v3.2", Name: "DeepSeek V3.2", Description: "Advanced general-purpose model designed with efficiency in mind.", OwnedBy: "DeepSeek", Reasoning: "toggle", ToolUse: false},
	{ID: "google/gemini-3-pro-preview", Name: "Gemini 3 Pro Preview", Description: "Preview of frontier-level fast model, distilled from Gemini 3 Pro and optimized for speed.", OwnedBy: "Google", Reasoning: "always", Vision: true, ReasoningEfforts: []string{"low", "high"}, DefaultEffort: "high"},
	{ID: "google/gemini-3-flash-preview", Name: "Gemini 3 Flash Preview", Description: "Preview of frontier-level fast model, distilled from Gemini 3 Pro and optimized for speed.", OwnedBy: "Google", Reasoning: "always", Vision: true, ReasoningEfforts: []string{"minimal", "low", "medium", "high"}, DefaultEffort: "medium"},
	{ID: "google/gemini-2.5-flash", Name: "Gemini 2.5 Flash", Description: "Low-latency, highly efficient model optimized for speed.", OwnedBy: "Google", Reasoning: "always", Vision: true, ReasoningEfforts: []string{"low", "medium", "high"}, DefaultEffort: "medium"},
	{ID: "google/gemini-2.5-flash-lite-preview-09-2025", Name: "Gemini 2.5 Flash Lite Preview", Description: "Lightweight variant of Gemini 2.5 Flash optimized for speed.", OwnedBy: "Google", Reasoning: "always", Vision: true, ReasoningEfforts: []string{"low", "medium", "high"}, DefaultEffort: "medium"},
	{ID: "google/gemini-2.5-flash-image", Name: "Nano Banana (Image)", Description: "Fast image generation model.", OwnedBy: "Google", Reasoning: "off", Vision: true, ToolUse: false},
	{ID: "moonshotai/kimi-k2.5", Name: "Kimi K2.5", Description: "SOTA open-weights model with exceptional EQ, coding, and agentic abilities.", OwnedBy: "Moonshot AI", Reasoning: "toggle", Vision: true},
	{ID: "moonshotai/kimi-k2-0905", Name: "Kimi K2", Description: "Older open-weights model with great EQ and coding abilities.", OwnedBy: "Moonshot AI", Reasoning: "toggle", ThinkingModelID: "moonshotai/kimi-k2-thinking"},
	{ID: "minimax/minimax-m2.5", Name: "MiniMax M2.5", Description: "Frontier open-weights coding model.", OwnedBy: "MiniMax", Reasoning: "off"},
	{ID: "minimax/minimax-m2.1", Name: "MiniMax M2.1", Description: "High-quality open-weights coding model.", OwnedBy: "MiniMax", Reasoning: "off"},
	{ID: "openai/gpt-5.2", Name: "GPT-5.2", Description: "The frontier flagship model delivering frontier general and STEM knowledge.", OwnedBy: "OpenAI", Reasoning: "always", Vision: true, ReasoningEfforts: []string{"low", "medium", "high", "xhigh"}, DefaultEffort: "medium"},
	{ID: "openai/gpt-5.1", Name: "GPT-5.1", Description: "The flagship model delivering strong general and emotional intelligence.", OwnedBy: "OpenAI", Reasoning: "always", Vision: true, ReasoningEfforts: []string{"low", "medium", "high"}, DefaultEffort: "medium"},
	{ID: "openai/gpt-oss-120b", Name: "GPT OSS 120B", Description: "High-performance open-weights model with exceptional STEM capabilities.", OwnedBy: "OpenAI", Reasoning: "always", ReasoningEfforts: []string{"low", "medium", "high"}, DefaultEffort: "medium"},
	{ID: "openai/gpt-5-mini", Name: "GPT-5 Mini", Description: "Streamlined version of GPT-5 optimized for lightweight tasks.", OwnedBy: "OpenAI", Reasoning: "always", ReasoningEfforts: []string{"low", "medium", "high"}, DefaultEffort: "medium"},
	{ID: "qwen/qwen3-vl-235b-a22b-instruct", Name: "Qwen 3 VL 235B A22B Instruct", Description: "Open-weight vision-language model excelling at document understanding and visual reasoning.", OwnedBy: "Qwen", Reasoning: "off", Vision: true},
	{ID: "qwen/qwen3-next-80b-a3b-instruct", Name: "Qwen 3 Next 80B A3B Instruct", Description: "Highly efficient experimental model that punches above its weight.", OwnedBy: "Qwen", Reasoning: "off"},
	{ID: "x-ai/grok-4.1-fast", Name: "Grok 4.1 Fast", Description: "Fast model with great agentic capabilities and limited censorship.", OwnedBy: "xAI", Reasoning: "toggle"},
	{ID: "z-ai/glm-5", Name: "GLM 5", Description: "Frontier open-weight model excelling at coding and math.", OwnedBy: "Z.ai", Reasoning: "toggle"},
	{ID: "z-ai/glm-4.7", Name: "GLM 4.7", Description: "High-quality open-weight model excelling at coding and math.", OwnedBy: "Z.ai", Reasoning: "toggle"},
	{ID: "z-ai/glm-4.7-flash", Name: "GLM 4.7 Flash", Description: "SOTA 30B-class model with excellent agentic capabilities.", OwnedBy: "Z.ai", Reasoning: "toggle"},
	{ID: "z-ai/glm-4.6", Name: "GLM 4.6", Description: "Reliable bilingual model for reasoning and tool use.", OwnedBy: "Z.ai", Reasoning: "toggle"},
}
var thinkingModelMap = map[string]string{
	"moonshotai/kimi-k2-0905": "moonshotai/kimi-k2-thinking",
}

func findModelInfo(modelID string) *ModelInfo {
	for i := range modelRegistry {
		if modelRegistry[i].ID == modelID {
			return &modelRegistry[i]
		}
	}
	return nil
}

func main() {
	loadConfig()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", corsMiddleware(authMiddleware(handleChatCompletions)))
	mux.HandleFunc("/v1/models", corsMiddleware(authMiddleware(handleModels)))
	mux.HandleFunc("/", corsMiddleware(handleRoot))
	server := &http.Server{
		Addr:              ":" + config.Port,
		Handler:           mux,
		ReadTimeout:       600 * time.Second,
		WriteTimeout:      600 * time.Second,
		ReadHeaderTimeout: 30 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	if len(config.ApiKeys) > 0 {
		slog.Info("鉴权已启用", "api_keys", len(config.ApiKeys))
	} else {
		slog.Warn("未配置 api_keys，接口无鉴权保护")
	}
	slog.Info("服务启动", "port", config.Port)
	if err := server.ListenAndServe(); err != nil {
		slog.Error("服务启动失败", "error", err)
		os.Exit(1)
	}
}

func getClientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		parts := strings.SplitN(xff, ",", 2)
		return strings.TrimSpace(parts[0])
	}
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}
	addr := r.RemoteAddr
	if idx := strings.LastIndex(addr, ":"); idx != -1 {
		return addr[:idx]
	}
	return addr
}

func generateUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
func corsMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next(w, r)
	}
}
func authMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(config.ApiKeys) == 0 {
			next(w, r)
			return
		}

		auth := r.Header.Get("Authorization")
		if auth == "" || !strings.HasPrefix(auth, "Bearer ") {
			slog.Warn("鉴权失败：缺少 Authorization 头", "ip", getClientIP(r), "path", r.URL.Path)
			sendError(w, "缺少 Authorization 头或格式错误，需要 Bearer token", "authentication_error", http.StatusUnauthorized)
			return
		}

		token := strings.TrimPrefix(auth, "Bearer ")
		valid := false
		for _, key := range config.ApiKeys {
			if key == token {
				valid = true
				break
			}
		}

		if !valid {
			slog.Warn("鉴权失败：无效的 API Key", "ip", getClientIP(r), "path", r.URL.Path)
			sendError(w, "无效的 API Key", "authentication_error", http.StatusUnauthorized)
			return
		}

		next(w, r)
	}
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	uptime := time.Since(stats.StartTime).Round(time.Second)
	totalReqs := atomic.LoadInt64(&stats.TotalRequests)
	successReqs := atomic.LoadInt64(&stats.SuccessRequests)
	inputTk := atomic.LoadInt64(&stats.TotalInputTokens)
	outputTk := atomic.LoadInt64(&stats.TotalOutputTokens)
	multimodal := atomic.LoadInt64(&stats.MultimodalCalls)

	successRate := float64(0)
	if totalReqs > 0 {
		successRate = float64(successReqs) / float64(totalReqs) * 100
	}
	var avgInput, avgOutput int64
	if successReqs > 0 {
		avgInput = inputTk / successReqs
		avgOutput = outputTk / successReqs
	}

	/* 快照模型使用统计（短暂加锁） */
	stats.mu.RLock()
	modelStatsMap := make(map[string]interface{}, len(stats.ModelUsage))
	for name, ms := range stats.ModelUsage {
		modelStatsMap[name] = map[string]interface{}{
			"requests":      ms.Requests,
			"input_tokens":  ms.InputTokens,
			"output_tokens": ms.OutputTokens,
		}
	}
	stats.mu.RUnlock()

	json.NewEncoder(w).Encode(map[string]interface{}{
		"version": "2.0.0",
		"telemetry": map[string]interface{}{
			"uptime":              uptime.String(),
			"total_calls":         totalReqs,
			"success_calls":       successReqs,
			"total_requests":      successReqs,
			"success_rate":        successRate,
			"rpm":                 stats.getRPM(),
			"total_input_tokens":  inputTk,
			"total_output_tokens": outputTk,
			"avg_input_tokens":    avgInput,
			"avg_output_tokens":   avgOutput,
			"multimodal_calls":    multimodal,
			"valid_tokens":        len(config.ApiKeys),
			"model_stats":         modelStatsMap,
		},
	})
}

func handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	var models []interface{}
	now := time.Now().Unix()

	for _, m := range modelRegistry {
		models = append(models, map[string]interface{}{
			"id":       m.ID,
			"object":   "model",
			"created":  now,
			"owned_by": m.OwnedBy,
		})
		if m.Reasoning == "toggle" {
			models = append(models, map[string]interface{}{
				"id":       m.ID + "-thinking",
				"object":   "model",
				"created":  now,
				"owned_by": m.OwnedBy,
			})
		}
		if len(m.ReasoningEfforts) > 0 {
			for _, effort := range m.ReasoningEfforts {
				models = append(models, map[string]interface{}{
					"id":       m.ID + "-thinking-" + effort,
					"object":   "model",
					"created":  now,
					"owned_by": m.OwnedBy,
				})
			}
		}
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}
func handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	clientIP := getClientIP(r)

	if r.Method != http.MethodPost {
		slog.Warn("无效请求方法", "method", r.Method, "ip", clientIP)
		sendError(w, "Method not allowed", "invalid_request_error", http.StatusMethodNotAllowed)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, config.MaxBodySize)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		sendError(w, "读取请求体失败", "invalid_request_error", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var reqMap map[string]interface{}
	if err := json.Unmarshal(body, &reqMap); err != nil {
		sendError(w, "无效的 JSON 格式", "invalid_request_error", http.StatusBadRequest)
		return
	}

	model, _ := reqMap["model"].(string)
	isStream, _ := reqMap["stream"].(bool)
	actualModel, enableReasoning, reasoningEffort := parseModelSuffix(model)
	if enableReasoning {
		if thinkingID, ok := thinkingModelMap[actualModel]; ok {
			reqMap["model"] = thinkingID
		} else {
			reqMap["model"] = actualModel
			if reasoningEffort != "" {
				reqMap["reasoning"] = map[string]interface{}{"effort": reasoningEffort}
			} else {
				reqMap["reasoning"] = map[string]interface{}{"enabled": true}
			}
		}
		delete(reqMap, "reasoning_effort")
	} else {
		convertReasoningField(reqMap)
	}
	hasTools := reqMap["tools"] != nil
	toolChoice, _ := reqMap["tool_choice"].(string)
	hasImages := detectImageContent(reqMap)

	slog.Info("收到请求",
		"ip", clientIP,
		"model", actualModel,
		"stream", isStream,
		"thinking", enableReasoning,
		"tools", hasTools,
		"tool_choice", toolChoice,
		"images", hasImages,
	)
	stats.recordRequest()
	if hasImages {
		stats.recordMultimodal()
	}

	upstreamBody, err := json.Marshal(reqMap)
	if err != nil {
		stats.recordFailure()
		sendError(w, "序列化请求失败", "server_error", http.StatusInternalServerError)
		return
	}

	upstreamReq, err := http.NewRequest(http.MethodPost, config.UpstreamURL, bytes.NewReader(upstreamBody))
	if err != nil {
		stats.recordFailure()
		sendError(w, "创建上游请求失败", "server_error", http.StatusInternalServerError)
		return
	}

	setUpstreamHeaders(upstreamReq)

	resp, err := httpClient.Do(upstreamReq)
	if err != nil {
		stats.recordFailure()
		slog.Error("上游请求失败", "ip", clientIP, "model", actualModel, "error", err)
		sendError(w, fmt.Sprintf("上游请求失败: %v", err), "server_error", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	/* 上游错误时，记录失败并返回标准 JSON 错误 */
	if resp.StatusCode != http.StatusOK {
		stats.recordFailure()
		respBody, _ := io.ReadAll(resp.Body)

		/* 日志中截断过长的响应体（如 HTML 页面），避免刷屏 */
		logBody := string(respBody)
		if len(logBody) > 500 {
			logBody = logBody[:500] + "...(truncated)"
		}
		slog.Error("上游返回错误",
			"ip", clientIP,
			"model", actualModel,
			"status", resp.StatusCode,
			"content_type", resp.Header.Get("Content-Type"),
			"response", logBody,
			"duration", time.Since(start).String(),
		)

		w.Header().Set("Content-Type", "application/json")

		/*
		 * 判断上游响应是否为有效 JSON
		 * 如果是 HTML（Vercel 安全检查页面等）或其他非 JSON 内容
		 * 统一返回标准 OpenAI 格式的错误响应
		 */
		contentType := resp.Header.Get("Content-Type")
		isJSON := strings.Contains(contentType, "application/json")

		if isJSON && len(respBody) > 0 {
			var testJSON map[string]interface{}
			if json.Unmarshal(respBody, &testJSON) == nil {
				w.WriteHeader(resp.StatusCode)
				w.Write(respBody)
				return
			}
		}

		/* 尝试从 SSE 包裹中提取错误信息 */
		if extracted := extractSSEJSON(respBody); extracted != nil {
			var errMap map[string]interface{}
			if json.Unmarshal(extracted, &errMap) == nil {
				if errObj, ok := errMap["error"].(map[string]interface{}); ok {
					msg, _ := errObj["message"].(string)
					code, _ := errObj["code"].(float64)
					if code == 0 {
						code = float64(resp.StatusCode)
					}
					sendError(w, msg, "upstream_error", int(code))
					return
				}
			}
		}

		/* 兜底：返回通用错误 */
		errMsg := fmt.Sprintf("上游返回错误: HTTP %d", resp.StatusCode)
		if strings.Contains(contentType, "text/html") {
			errMsg = fmt.Sprintf("上游返回安全验证页面 (HTTP %d)，请稍后重试", resp.StatusCode)
		}
		sendError(w, errMsg, "upstream_error", resp.StatusCode)
		return
	}

	var inputTokens, outputTokens int64
	var success bool
	if isStream {
		inputTokens, outputTokens, success = handleStreamResponse(w, resp)
	} else {
		inputTokens, outputTokens, success = handleNonStreamResponse(w, resp)
	}
	if success {
		stats.recordSuccess(actualModel, inputTokens, outputTokens)
	} else {
		stats.recordFailure()
	}

	slog.Info("请求完成",
		"ip", clientIP,
		"model", actualModel,
		"stream", isStream,
		"input_tokens", inputTokens,
		"output_tokens", outputTokens,
		"duration", time.Since(start).String(),
	)
}

func parseModelSuffix(model string) (string, bool, string) {
	suffixes := []struct {
		suffix string
		effort string
	}{
		{"-thinking-xhigh", "xhigh"},
		{"-thinking-high", "high"},
		{"-thinking-medium", "medium"},
		{"-thinking-low", "low"},
		{"-thinking-minimal", "minimal"},
		{"-thinking", ""},
	}

	for _, s := range suffixes {
		if strings.HasSuffix(model, s.suffix) {
			actualModel := strings.TrimSuffix(model, s.suffix)
			return actualModel, true, s.effort
		}
	}

	return model, false, ""
}
func detectImageContent(reqMap map[string]interface{}) bool {
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return false
	}
	for _, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		contentArr, ok := msgMap["content"].([]interface{})
		if !ok {
			continue
		}
		for _, part := range contentArr {
			partMap, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			if partType, _ := partMap["type"].(string); partType == "image_url" {
				return true
			}
		}
	}
	return false
}
func convertReasoningField(reqMap map[string]interface{}) {
	if effort, ok := reqMap["reasoning_effort"].(string); ok {
		reqMap["reasoning"] = map[string]interface{}{
			"effort": effort,
		}
		delete(reqMap, "reasoning_effort")
	}
}

func setUpstreamHeaders(req *http.Request) {
	sessionID := generateUUID()
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36 Edg/144.0.0.0")
	req.Header.Set("Accept", "*/*")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6")
	req.Header.Set("Cache-Control", "no-cache")
	req.Header.Set("DNT", "1")
	req.Header.Set("Origin", "https://libreassistant.vercel.app")
	req.Header.Set("Pragma", "no-cache")
	req.Header.Set("Priority", "u=1, i")
	req.Header.Set("Referer", fmt.Sprintf("https://libreassistant.vercel.app/%s", sessionID))
	req.Header.Set("Sec-CH-UA", `"Not(A:Brand";v="8", "Chromium";v="144", "Microsoft Edge";v="144"`)
	req.Header.Set("Sec-CH-UA-Mobile", "?0")
	req.Header.Set("Sec-CH-UA-Platform", `"Windows"`)
	req.Header.Set("Sec-Fetch-Dest", "empty")
	req.Header.Set("Sec-Fetch-Mode", "cors")
	req.Header.Set("Sec-Fetch-Site", "same-origin")
}
func handleStreamResponse(w http.ResponseWriter, resp *http.Response) (int64, int64, bool) {
	var inputTokens, outputTokens int64

	flusher, ok := w.(http.Flusher)
	if !ok {
		sendError(w, "不支持流式传输", "server_error", http.StatusInternalServerError)
		return 0, 0, false
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, config.ScannerBufSize), config.ScannerBufSize)

	hasData := false
	for scanner.Scan() {
		line := scanner.Text()

		if line == "" {
			continue
		}

		if line == "data: [DONE]" {
			fmt.Fprintf(w, "data: [DONE]\n\n")
			flusher.Flush()
			hasData = true
			break
		}

		if strings.HasPrefix(line, "data: ") {
			hasData = true
			dataStr := strings.TrimPrefix(line, "data: ")
			var raw map[string]interface{}
			if json.Unmarshal([]byte(dataStr), &raw) == nil {
				if usage, ok := raw["usage"].(map[string]interface{}); ok {
					if pt, ok := usage["prompt_tokens"].(float64); ok {
						inputTokens = int64(pt)
					}
					if ct, ok := usage["completion_tokens"].(float64); ok {
						outputTokens = int64(ct)
					}
				}
			}

			converted := convertStreamChunk(dataStr)
			fmt.Fprintf(w, "data: %s\n\n", converted)
			flusher.Flush()
		}
	}

	/* 上游未返回任何有效 SSE 数据，说明响应格式异常 */
	if !hasData {
		slog.Warn("上游流式响应无有效 SSE 数据")
		return 0, 0, false
	}
	return inputTokens, outputTokens, true
}
func convertStreamChunk(dataStr string) string {
	var chunk map[string]interface{}
	if err := json.Unmarshal([]byte(dataStr), &chunk); err != nil {
		return dataStr
	}

	choices, ok := chunk["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		result, _ := json.Marshal(chunk)
		return string(result)
	}

	for _, choice := range choices {
		choiceMap, ok := choice.(map[string]interface{})
		if !ok {
			continue
		}
		delete(choiceMap, "native_finish_reason")

		delta, ok := choiceMap["delta"].(map[string]interface{})
		if !ok {
			continue
		}

		/*
		 * 从 reasoning_details 中提取并拼接所有思考文本
		 * 跳过 reasoning.encrypted 类型（加密的推理数据，不可用）
		 */
		var reasoningParts []string
		if reasoningDetails, ok := delta["reasoning_details"].([]interface{}); ok {
			for _, detail := range reasoningDetails {
				detailMap, ok := detail.(map[string]interface{})
				if !ok {
					continue
				}
				detailType, _ := detailMap["type"].(string)
				if detailType != "reasoning.encrypted" {
					if text, ok := detailMap["text"].(string); ok && text != "" {
						reasoningParts = append(reasoningParts, text)
					}
				}
			}
		}
		if len(reasoningParts) > 0 {
			delta["reasoning_content"] = strings.Join(reasoningParts, "")
		} else if reasoning, ok := delta["reasoning"].(string); ok && reasoning != "" {
			delta["reasoning_content"] = reasoning
		}
		delete(delta, "reasoning")
		delete(delta, "reasoning_details")
		delete(delta, "annotations")
	}
	delete(chunk, "provider")
	cleanUsage(chunk)

	result, err := json.Marshal(chunk)
	if err != nil {
		return dataStr
	}
	return string(result)
}

/*
 * extractSSEJSON 从 SSE 格式的响应体中提取第一个有效 JSON 数据
 * 上游有时即使客户端请求 stream=false，也会返回 SSE 格式包裹的内容
 */
func extractSSEJSON(body []byte) []byte {
	lines := strings.Split(string(body), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				continue
			}
			return []byte(data)
		}
	}
	return nil
}

func handleNonStreamResponse(w http.ResponseWriter, resp *http.Response) (int64, int64, bool) {
	var inputTokens, outputTokens int64

	w.Header().Set("Content-Type", "application/json")

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		sendError(w, "读取上游响应失败", "server_error", http.StatusBadGateway)
		return 0, 0, false
	}

	var respMap map[string]interface{}
	if err := json.Unmarshal(body, &respMap); err != nil {
		/*
		 * 上游可能以 SSE 格式返回（即使请求了 stream=false）
		 * 尝试从 "data: {...}" 行中提取 JSON
		 */
		extracted := extractSSEJSON(body)
		if extracted != nil {
			if err2 := json.Unmarshal(extracted, &respMap); err2 != nil {
				slog.Warn("上游返回非 JSON 响应", "body", string(body))
				sendError(w, fmt.Sprintf("上游返回非 JSON 响应: %s", string(body)), "upstream_error", http.StatusBadGateway)
				return 0, 0, false
			}
			/* 检测是否为上游错误响应 */
			if errObj, ok := respMap["error"].(map[string]interface{}); ok {
				msg, _ := errObj["message"].(string)
				code, _ := errObj["code"].(float64)
				if code == 0 {
					code = 502
				}
				slog.Warn("上游返回错误（SSE 包裹）", "message", msg, "code", int(code))
				sendError(w, msg, "upstream_error", int(code))
				return 0, 0, false
			}
		} else {
			slog.Warn("上游返回非 JSON 响应", "body", string(body))
			sendError(w, fmt.Sprintf("上游返回非 JSON 响应: %s", string(body)), "upstream_error", http.StatusBadGateway)
			return 0, 0, false
		}
	}

	/* 提取 token 用量 */
	if usage, ok := respMap["usage"].(map[string]interface{}); ok {
		if pt, ok := usage["prompt_tokens"].(float64); ok {
			inputTokens = int64(pt)
		}
		if ct, ok := usage["completion_tokens"].(float64); ok {
			outputTokens = int64(ct)
		}
	}
	convertResponseReasoning(respMap)
	if converted, err := json.Marshal(respMap); err == nil {
		body = converted
	}

	w.WriteHeader(http.StatusOK)
	w.Write(body)
	return inputTokens, outputTokens, true
}
func convertResponseReasoning(respMap map[string]interface{}) {
	delete(respMap, "provider")
	cleanUsage(respMap)

	choices, ok := respMap["choices"].([]interface{})
	if !ok {
		return
	}
	for _, choice := range choices {
		choiceMap, ok := choice.(map[string]interface{})
		if !ok {
			continue
		}
		delete(choiceMap, "native_finish_reason")
		message, ok := choiceMap["message"].(map[string]interface{})
		if !ok {
			continue
		}

		/*
		 * 从 reasoning_details 中提取并拼接所有思考文本
		 * 跳过 reasoning.encrypted 类型（加密的推理数据，不可用）
		 */
		var reasoningParts []string
		if reasoningDetails, ok := message["reasoning_details"].([]interface{}); ok {
			for _, detail := range reasoningDetails {
				detailMap, ok := detail.(map[string]interface{})
				if !ok {
					continue
				}
				detailType, _ := detailMap["type"].(string)
				if detailType != "reasoning.encrypted" {
					if text, ok := detailMap["text"].(string); ok && text != "" {
						reasoningParts = append(reasoningParts, text)
					}
				}
			}
		}

		/* 优先使用 reasoning_details 拼接结果，回退到 reasoning 字段 */
		if len(reasoningParts) > 0 {
			message["reasoning_content"] = strings.Join(reasoningParts, "")
		} else if reasoning, ok := message["reasoning"].(string); ok && reasoning != "" {
			message["reasoning_content"] = reasoning
		}
		delete(message, "reasoning")
		delete(message, "reasoning_details")
		delete(message, "annotations")
		delete(message, "refusal")
	}
}

/*
 * cleanUsage 清理 usage 中的非 OpenAI 标准字段
 * 保留：prompt_tokens, completion_tokens, total_tokens, completion_tokens_details
 * 移除：cost, is_byok, cost_details, prompt_tokens_details 等上游附加字段
 */
func cleanUsage(respMap map[string]interface{}) {
	usage, ok := respMap["usage"].(map[string]interface{})
	if !ok {
		return
	}
	delete(usage, "cost")
	delete(usage, "is_byok")
	delete(usage, "cost_details")
	delete(usage, "prompt_tokens_details")
}
func sendError(w http.ResponseWriter, message string, errType string, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    errType,
			"code":    status,
		},
	})
}
