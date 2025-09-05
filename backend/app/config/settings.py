"""
配置管理系统

基于Pydantic的配置管理，支持环境变量覆盖和YAML配置文件
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from pydantic import BaseSettings, Field, validator


class WebSocketSettings(BaseSettings):
    """WebSocket配置"""
    ping_interval: int = 30
    ping_timeout: int = 10
    max_reconnect_attempts: int = 5
    heartbeat_interval: int = 60


class SecuritySettings(BaseSettings):
    """安全配置"""
    max_request_size: int = 10485760  # 10MB
    rate_limit_per_minute: int = 60
    jwt_expire_minutes: int = 1440  # 24小时
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = "HS256"


class SystemSettings(BaseSettings):
    """系统配置"""
    max_context_tokens: int = 8192
    session_timeout: int = 3600
    max_sessions_per_user: int = 10
    websocket: WebSocketSettings = WebSocketSettings()
    security: SecuritySettings = SecuritySettings()


class AIModelSettings(BaseSettings):
    """AI模型配置"""
    provider: str = "openai"  # openai, anthropic, local
    name: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    frequency_penalty: float = 0.1
    presence_penalty: float = 0.1


class StreamingSettings(BaseSettings):
    """流式输出配置"""
    enabled: bool = True
    chunk_size: int = 16
    delay_ms: int = 50
    buffer_size: int = 1024


class ContextSettings(BaseSettings):
    """上下文管理配置"""
    max_history_messages: int = 50
    summary_trigger_tokens: int = 6000
    importance_threshold: float = 0.6


class VoiceSettings(BaseSettings):
    """语音处理配置"""
    asr_model: str = "whisper-medium"
    tts_provider: str = "azure"
    tts_voice: str = "zh-CN-XiaoxiaoNeural"
    sample_rate: int = 16000
    chunk_duration: int = 30


class AISettings(BaseSettings):
    """AI相关配置"""
    model: AIModelSettings = AIModelSettings()
    streaming: StreamingSettings = StreamingSettings()
    context: ContextSettings = ContextSettings()
    voice: VoiceSettings = VoiceSettings()


class CharacterPersonality(BaseSettings):
    """角色性格配置"""
    traits: List[str] = ["友善", "耐心", "专业", "幽默"]
    introverted: bool = False
    formality_level: float = 0.6


class WritingStyle(BaseSettings):
    """写作风格配置"""
    tone: str = "温和友善"
    sentence_length: str = "中等"
    use_emoji: bool = True
    technical_level: str = "适中"


class DefaultCharacter(BaseSettings):
    """默认角色配置"""
    name: str = "小助手"
    personality: CharacterPersonality = CharacterPersonality()
    knowledge_domains: List[str] = ["通用知识", "技术支持", "日常对话"]
    writing_style: WritingStyle = WritingStyle()


class ConsistencySettings(BaseSettings):
    """一致性检查配置"""
    check_enabled: bool = True
    deviation_threshold: float = 0.7
    adjustment_attempts: int = 2


class CharacterSettings(BaseSettings):
    """人设系统配置"""
    default: DefaultCharacter = DefaultCharacter()
    consistency: ConsistencySettings = ConsistencySettings()


class MemorySettings(BaseSettings):
    """记忆系统配置"""
    # 短期记忆
    short_term_max_tokens: int = 2000
    short_term_retention_hours: int = 24
    
    # 中期记忆
    medium_term_max_entries: int = 100
    medium_term_retention_days: int = 30
    medium_term_summary_min_length: int = 50
    
    # 长期记忆
    long_term_update_frequency: int = 300
    long_term_relevance_threshold: float = 0.5
    long_term_max_entities_per_user: int = 1000
    
    # 记忆检索
    retrieval_max_results: int = 10
    retrieval_similarity_threshold: float = 0.7
    retrieval_boost_recent: bool = True


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    # PostgreSQL
    postgres_url: str = Field(..., env="DATABASE_URL")
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20
    postgres_pool_timeout: int = 30
    postgres_echo: bool = False
    
    # Redis
    redis_url: str = Field(..., env="REDIS_URL")
    redis_max_connections: int = 50
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    redis_retry_on_timeout: bool = True


class MediaSettings(BaseSettings):
    """媒体处理配置"""
    # 图像
    image_max_size: int = 5242880  # 5MB
    image_allowed_formats: List[str] = ["jpg", "jpeg", "png", "gif", "webp"]
    image_ocr_enabled: bool = True
    image_vision_analysis: bool = True
    
    # 音频
    audio_max_duration: int = 300  # 5分钟
    audio_allowed_formats: List[str] = ["wav", "mp3", "m4a", "ogg"]
    audio_auto_transcribe: bool = True
    
    # 文件上传
    upload_temp_dir: str = "/tmp/chatbot_uploads"
    upload_cleanup_interval: int = 3600


class Live2DSettings(BaseSettings):
    """Live2D配置"""
    enabled: bool = True
    models_path: str = "/static/live2d/models"
    default_model: str = "default_character"
    
    # 动作配置
    idle_motions: List[str] = ["idle_01", "idle_02", "idle_03"]
    talk_motions: List[str] = ["talk_01", "talk_02"]
    emotion_motions: Dict[str, List[str]] = {
        "happy": ["happy_01", "happy_02"],
        "sad": ["sad_01"],
        "surprised": ["surprised_01"],
        "angry": ["angry_01"]
    }
    
    # 表情检测
    emotion_detection_enabled: bool = True
    emotion_confidence_threshold: float = 0.6
    emotion_update_interval: int = 2000  # 毫秒


class MonitoringSettings(BaseSettings):
    """监控配置"""
    # 日志
    log_level: str = "INFO"
    log_format: str = "json"
    log_file_rotation: str = "1 day"
    log_max_file_size: str = "100MB"
    
    # 性能监控
    metrics_enabled: bool = True
    metrics_collection_interval: int = 60
    metrics_retention_days: int = 7
    
    # 健康检查
    health_check_interval: int = 30
    health_check_timeout: int = 10


class Settings(BaseSettings):
    """主配置类"""
    
    # 环境配置
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # API配置
    api_title: str = "AI聊天机器人系统"
    api_version: str = "1.0.0"
    api_description: str = "智能AI聊天机器人系统后端API"
    
    # CORS配置
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    cors_allow_headers: List[str] = ["*"]
    
    # 第三方API密钥
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    azure_speech_key: Optional[str] = Field(None, env="AZURE_SPEECH_KEY")
    azure_speech_region: Optional[str] = Field(None, env="AZURE_SPEECH_REGION")
    
    # 各模块配置
    system: SystemSettings = SystemSettings()
    ai: AISettings = AISettings()
    character: CharacterSettings = CharacterSettings()
    memory: MemorySettings = MemorySettings()
    database: DatabaseSettings = DatabaseSettings()
    media: MediaSettings = MediaSettings()
    live2d: Live2DSettings = Live2DSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """初始化配置，支持从YAML文件加载"""
        super().__init__(**kwargs)
        self._load_yaml_config()
    
    def _load_yaml_config(self):
        """从YAML文件加载配置"""
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f)
                
                # 根据环境选择配置
                env_config = yaml_config.get(self.environment, {})
                base_config = {k: v for k, v in yaml_config.items() 
                             if k not in ["development", "production"]}
                
                # 合并配置
                merged_config = {**base_config, **env_config}
                
                # 更新配置值
                self._update_from_dict(merged_config)
                
            except Exception as e:
                print(f"警告: 无法加载YAML配置文件: {e}")
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_attr = getattr(self, key)
                if hasattr(current_attr, '__dict__'):
                    # 如果是嵌套的配置对象，递归更新
                    self._update_nested_config(current_attr, value)
                else:
                    # 直接设置值
                    setattr(self, key, value)
    
    def _update_nested_config(self, config_obj: Any, new_values: Dict[str, Any]):
        """更新嵌套配置对象"""
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                if hasattr(config_obj, key):
                    current_attr = getattr(config_obj, key)
                    if hasattr(current_attr, '__dict__') and isinstance(value, dict):
                        self._update_nested_config(current_attr, value)
                    else:
                        setattr(config_obj, key, value)
    
    @validator("environment")
    def validate_environment(cls, v):
        """验证环境配置"""
        if v not in ["development", "production", "testing"]:
            raise ValueError("环境必须是 development, production 或 testing")
        return v
    
    @validator("cors_origins")
    def validate_cors_origins(cls, v):
        """验证CORS配置"""
        if not isinstance(v, list):
            raise ValueError("CORS origins必须是列表")
        return v
    
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment == "production"
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """获取所有API密钥"""
        return {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "azure_speech": self.azure_speech_key,
        }


@lru_cache()
def get_settings() -> Settings:
    """获取配置实例（单例模式）"""
    return Settings()


# 导出配置实例
settings = get_settings()
