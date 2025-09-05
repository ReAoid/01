# AI聊天机器人系统

一个功能完整的AI聊天机器人系统，支持实时对话、多模态交互、虚拟形象和长期记忆功能。

## 🌟 主要特性

### 核心功能
- **智能对话**: 支持OpenAI GPT、Ollama本地模型等多种AI提供商
- **流式响应**: WebSocket实时流式输出，模拟真人打字
- **多轮对话**: 智能上下文管理和对话历史
- **人设系统**: 多种AI角色，支持个性化定制
- **长期记忆**: 用户偏好学习和对话总结
- **本地部署**: 支持Ollama本地AI模型，数据完全私有

### 多模态交互
- **语音对话**: 语音输入(ASR)和语音输出(TTS)
- **图像理解**: 图片上传和OCR识别
- **文件处理**: 支持多种文档格式
- **Live2D虚拟形象**: 交互式虚拟角色

### 技术特点
- **现代化架构**: FastAPI后端 + React前端
- **实时通信**: WebSocket双向通信
- **响应式设计**: 适配桌面和移动设备
- **PWA支持**: 可安装的Web应用
- **Docker部署**: 一键容器化部署

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React前端     │    │  FastAPI后端    │    │   数据库层      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • 聊天界面      │    │ • AI流水线      │    │ • PostgreSQL    │
│ • WebSocket客户端│◄──►│ • 上下文管理    │◄──►│ • Redis缓存     │
│ • 语音处理      │    │ • 记忆系统      │    │ • 向量数据库    │
│ • Live2D渲染    │    │ • 流式处理      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose (推荐)
- PostgreSQL 15+
- Redis 7+
- Ollama (可选，用于本地AI模型)

### 方式一：Docker部署 (推荐)

1. **克隆项目**
```bash
git clone <repository-url>
cd ai-chatbot-system
```

2. **配置环境变量**
```bash
# 复制环境配置文件
cp backend/env.example backend/.env

# 编辑配置文件，填入你的API密钥
nano backend/.env
```

3. **启动服务**
```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
```

4. **访问应用**
- 前端界面: http://localhost:3000
- 后端API: http://localhost:8000
- API文档: http://localhost:8000/docs

### 方式二：本地开发

1. **后端设置**
```bash
# 创建虚拟环境
conda create --name 01 python=3.11
activate 01
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖（推荐使用自动安装脚本）
python ../install_dependencies.py
# 或手动安装
pip install -r requirements.txt

# 配置环境变量
cp env.example .env
# 编辑 .env 文件

# 启动后端服务
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **前端设置**
```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

3. **数据库设置**
```bash
# 启动PostgreSQL和Redis
docker-compose up -d postgres redis

# 或使用本地安装的数据库
```

### 方式三：Ollama本地AI模型部署

如果您希望使用本地AI模型而不依赖外部API，可以配置Ollama：

1. **安装Ollama**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: 从 https://ollama.ai/download 下载安装包
```

2. **启动Ollama服务**
```bash
ollama serve
```

3. **拉取推荐模型**
```bash
# 中文对话模型
ollama pull qwen:7b

# 通用模型
ollama pull llama2:7b

# 代码生成模型
ollama pull codellama:7b
```

4. **配置系统使用Ollama**
```bash
# 编辑 .env 文件
OLLAMA_BASE_URL=http://localhost:11434

# 或修改 backend/app/config/config.yaml
ai:
  model:
    provider: "ollama"
    name: "qwen:7b"
```

5. **测试Ollama集成**
```bash
# 环境检查（推荐先运行）
python check_environment.py

# 完整测试
python test_ollama.py

# 或使用pytest
pytest test_ollama_simple.py -v -s
```

> **遇到问题？** 查看 [故障排除指南](docs/troubleshooting.md) 获取详细的解决方案。
> 
> **常见问题快速修复**：
> - 缺少依赖：`python install_dependencies.py`
> - pytest异步错误：`pip install pytest-asyncio` 或直接运行 `python test_ollama.py`
> - Ollama连接失败：确保运行 `ollama serve`

## 📁 项目结构

```
ai-chatbot-system/
├── backend/                    # 后端服务
│   ├── app/                   # 主应用程序
│   │   ├── config/           # 配置管理
│   │   ├── core/             # 核心功能(AI流水线)
│   │   ├── models/           # 数据模型
│   │   ├── api/              # API路由
│   │   ├── services/         # 业务服务
│   │   └── utils/            # 工具函数
│   ├── requirements.txt      # Python依赖
│   └── Dockerfile           # Docker配置
├── frontend/                  # 前端应用
│   ├── src/                  # 源代码
│   │   ├── components/       # React组件
│   │   ├── hooks/           # 自定义Hooks
│   │   ├── services/        # API服务
│   │   ├── store/           # 状态管理
│   │   └── utils/           # 工具函数
│   ├── package.json         # Node.js依赖
│   └── vite.config.ts      # Vite配置
├── docs/                     # 文档
├── docker-compose.yml       # Docker Compose配置
└── README.md               # 项目说明
```

## ⚙️ 配置说明

### 后端配置 (`backend/.env`)

```bash
# 基础配置
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key

# AI服务配置
OPENAI_API_KEY=sk-your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# 数据库配置
DATABASE_URL=postgresql://user:pass@localhost:5432/chatbot_db
REDIS_URL=redis://localhost:6379/0

# Azure语音服务
AZURE_SPEECH_KEY=your-azure-speech-key
AZURE_SPEECH_REGION=your-region
```

### 前端配置 (`frontend/.env`)

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## 🎯 功能详解

### 1. AI对话流水线

系统实现了完整的AI对话处理流水线：

1. **输入预处理**: 文本清理、标准化
2. **意图识别**: 用户意图和情感分析
3. **上下文管理**: 智能上下文裁剪和会话摘要
4. **记忆检索**: 长期记忆和用户偏好匹配
5. **人设应用**: 角色一致性检查和调整
6. **AI生成**: 流式响应生成
7. **后处理**: 内容过滤和格式化
8. **记忆更新**: 新记忆提取和存储

### 2. 多角色人设系统

- **默认助手**: 友善专业的通用助手
- **专业顾问**: 严谨高效的商务顾问  
- **活泼助手**: 可爱热情的年轻助手
- **自定义角色**: 支持用户自定义人设

### 3. 长期记忆系统

- **事实记忆**: 用户基本信息和偏好
- **关系记忆**: 人际关系和社交网络
- **事件记忆**: 重要事件和时间节点
- **情感记忆**: 情感状态和心情变化

### 4. 多模态交互

- **语音输入**: Whisper ASR语音识别
- **语音输出**: Azure TTS文本转语音
- **图像处理**: OCR文字识别和图像理解
- **文件上传**: PDF、Word等文档处理

## 🔧 开发指南

### API文档

启动后端服务后，访问以下地址查看API文档：

- **交互式API文档**: http://localhost:8000/docs
- **ReDoc文档**: http://localhost:8000/redoc
- **系统状态**: http://localhost:8000/health
- **Ollama管理**: http://localhost:8000/api/ollama/health

#### Ollama API示例

```bash
# 检查Ollama服务状态
curl http://localhost:8000/api/ollama/health

# 获取可用模型列表
curl http://localhost:8000/api/ollama/models

# 拉取新模型
curl -X POST http://localhost:8000/api/ollama/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model_name": "mistral:7b"}'

# 测试模型对话
curl -X POST http://localhost:8000/api/ollama/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen:7b",
    "messages": [{"role": "user", "content": "你好"}],
    "temperature": 0.7
  }'
```

### WebSocket接口

```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/chat/session-id');

// 发送消息
ws.send(JSON.stringify({
  type: 'message',
  content: '你好！'
}));

// 接收消息
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('收到消息:', data);
};
```

### 自定义角色

```python
# 在 backend/app/core/character_system.py 中添加新角色
new_character = {
    "id": "custom",
    "name": "自定义角色",
    "description": "角色描述",
    "personality": {
        "traits": ["特征1", "特征2"],
        "formality_level": 0.7
    },
    "writing_style": {
        "tone": "语调风格",
        "use_emoji": True
    }
}
```

## 🧪 测试

### 后端测试
```bash
cd backend
pytest tests/ -v
```

### 前端测试
```bash
cd frontend
npm run test
```

### 集成测试
```bash
# 启动所有服务
docker-compose up -d

# 运行端到端测试
npm run test:e2e
```

## 📊 监控和日志

### 日志查看
```bash
# 查看后端日志
docker-compose logs -f backend

# 查看所有服务日志
docker-compose logs -f
```

### 性能监控

启用监控服务：
```bash
docker-compose --profile monitoring up -d
```

访问监控面板：
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (admin/admin)

## 🔒 安全考虑

- API密钥安全存储
- 输入内容过滤和验证
- 速率限制和防护
- HTTPS加密传输
- 用户数据隐私保护

## 🚀 部署到生产环境

### 1. 服务器要求
- CPU: 4核心以上
- 内存: 8GB以上
- 存储: 50GB以上SSD
- 网络: 稳定的互联网连接

### 2. 域名和SSL配置
```bash
# 配置Nginx SSL
cp nginx/ssl.example/* nginx/ssl/
# 编辑 nginx/conf.d/default.conf
```

### 3. 环境变量配置
```bash
# 生产环境配置
ENVIRONMENT=production
DEBUG=false
# 使用强密钥
SECRET_KEY=your-production-secret-key
```

### 4. 数据库优化
```bash
# PostgreSQL优化配置
# 编辑 postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持和反馈

- 🐛 [报告Bug](https://github.com/your-repo/issues)
- 💡 [功能建议](https://github.com/your-repo/issues)
- 📧 邮件支持: support@example.com
- 📚 [详细文档](./docs/)

## 🙏 致谢

感谢以下开源项目：
- [FastAPI](https://fastapi.tiangolo.com/)
- [React](https://reactjs.org/)
- [OpenAI](https://openai.com/)
- [Live2D](https://www.live2d.com/)

---

⭐ 如果这个项目对你有帮助，请给它一个星标！