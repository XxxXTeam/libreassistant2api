# LibreAssistant2API

将 [LibreAssistant](https://libreassistant.vercel.app) 转换为 OpenAI 兼容的 API 代理。

## 功能

- OpenAI 兼容的 `/v1/chat/completions` 接口
- 支持流式和非流式响应
- 支持思考模式（reasoning）
- 支持工具调用（function calling）
- 支持图片输入
- API Key 鉴权
- 实时统计（RPM / TPM）
- 上游错误智能处理（SSE 包裹、HTML 安全页面等）

## 配置

创建 `config.json`：

```json
{
  "api_keys": [],
  "port": "8080",
  "upstream_url": "https://libreassistant.vercel.app/api/ai",
  "max_body_size": 104857600,
  "scanner_buf_size": 10485760
}
```

- **api_keys**: API 密钥列表，为空则不鉴权
- **port**: 监听端口
- **upstream_url**: 上游 API 地址
- **max_body_size**: 最大请求体大小（字节）
- **scanner_buf_size**: 流式扫描缓冲区大小（字节）

## 构建

```bash
# Linux AMD64
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-s -w" -trimpath -o app .

# Windows
go build -o libreassistant2api.exe .
```

## 运行

```bash
./app
```

## API 端点

- `GET /` — 服务状态和统计信息
- `GET /v1/models` — 模型列表
- `POST /v1/chat/completions` — 聊天补全
