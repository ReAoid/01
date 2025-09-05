"""
用户相关数据模型

定义用户、用户偏好等相关的Pydantic模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, EmailStr


class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELETED = "deleted"


class UserRole(str, Enum):
    """用户角色枚举"""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"


class ThemePreference(str, Enum):
    """主题偏好枚举"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class LanguagePreference(str, Enum):
    """语言偏好枚举"""
    ZH_CN = "zh-CN"
    ZH_TW = "zh-TW"
    EN_US = "en-US"
    JA_JP = "ja-JP"


class VoiceSettings(BaseModel):
    """语音设置"""
    enabled: bool = True
    voice_id: str = "zh-CN-XiaoxiaoNeural"
    speed: float = 1.0  # 语速倍率
    pitch: float = 1.0  # 音调倍率
    volume: float = 1.0  # 音量倍率
    auto_play: bool = False  # 自动播放AI回复


class Live2DSettings(BaseModel):
    """Live2D设置"""
    enabled: bool = True
    character_model: str = "default_character"
    animation_speed: float = 1.0
    interaction_enabled: bool = True
    emotion_response: bool = True
    idle_animations: bool = True


class ChatSettings(BaseModel):
    """聊天设置"""
    send_on_enter: bool = True  # Enter发送消息
    show_typing_indicator: bool = True
    show_timestamps: bool = True
    message_sound: bool = True
    stream_response: bool = True
    auto_scroll: bool = True
    max_context_messages: int = 20


class PrivacySettings(BaseModel):
    """隐私设置"""
    save_chat_history: bool = True
    analytics_enabled: bool = True
    personalization_enabled: bool = True
    data_sharing_enabled: bool = False


class UserPreferences(BaseModel):
    """用户偏好设置"""
    theme: ThemePreference = ThemePreference.AUTO
    language: LanguagePreference = LanguagePreference.ZH_CN
    timezone: str = "Asia/Shanghai"
    
    # 功能设置
    voice: VoiceSettings = VoiceSettings()
    live2d: Live2DSettings = Live2DSettings()
    chat: ChatSettings = ChatSettings()
    privacy: PrivacySettings = PrivacySettings()
    
    # AI设置
    preferred_ai_model: Optional[str] = None
    default_temperature: float = 0.7
    default_character: str = "default"
    
    # 自定义设置
    custom_settings: Optional[Dict[str, Any]] = None


class UserProfile(BaseModel):
    """用户档案"""
    nickname: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    interests: Optional[List[str]] = None
    occupation: Optional[str] = None
    location: Optional[str] = None
    birth_date: Optional[datetime] = None


class UserStats(BaseModel):
    """用户统计信息"""
    total_sessions: int = 0
    total_messages: int = 0
    total_chat_time: int = 0  # 总聊天时间（秒）
    favorite_characters: List[str] = []
    most_active_hours: List[int] = []
    join_date: datetime
    last_active: Optional[datetime] = None


class User(BaseModel):
    """用户模型"""
    id: UUID = Field(default_factory=uuid4)
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    hashed_password: str
    
    # 基本信息
    status: UserStatus = UserStatus.ACTIVE
    role: UserRole = UserRole.USER
    is_verified: bool = False
    
    # 用户档案
    profile: Optional[UserProfile] = None
    preferences: UserPreferences = UserPreferences()
    stats: Optional[UserStats] = None
    
    # 时间戳
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    verified_at: Optional[datetime] = None
    
    # 订阅信息
    subscription_tier: str = "free"  # free, premium, enterprise
    subscription_expires: Optional[datetime] = None
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    @validator("username")
    def validate_username(cls, v):
        """验证用户名"""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("用户名只能包含字母、数字、下划线和连字符")
        return v.lower()


class UserCreate(BaseModel):
    """创建用户请求模型"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)
    
    # 可选的初始设置
    nickname: Optional[str] = None
    language: LanguagePreference = LanguagePreference.ZH_CN
    timezone: str = "Asia/Shanghai"
    
    @validator("username")
    def validate_username(cls, v):
        """验证用户名"""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("用户名只能包含字母、数字、下划线和连字符")
        return v.lower()
    
    @validator("password")
    def validate_password(cls, v):
        """验证密码强度"""
        if len(v) < 8:
            raise ValueError("密码长度至少8个字符")
        if not any(c.isupper() for c in v):
            raise ValueError("密码必须包含至少一个大写字母")
        if not any(c.islower() for c in v):
            raise ValueError("密码必须包含至少一个小写字母")
        if not any(c.isdigit() for c in v):
            raise ValueError("密码必须包含至少一个数字")
        return v


class UserResponse(BaseModel):
    """用户响应模型"""
    id: UUID
    email: EmailStr
    username: str
    status: UserStatus
    role: UserRole
    is_verified: bool
    
    profile: Optional[UserProfile] = None
    preferences: UserPreferences
    stats: Optional[UserStats] = None
    
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    
    subscription_tier: str
    subscription_expires: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """更新用户请求模型"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    profile: Optional[UserProfile] = None
    preferences: Optional[UserPreferences] = None
    
    @validator("username")
    def validate_username(cls, v):
        """验证用户名"""
        if v and not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("用户名只能包含字母、数字、下划线和连字符")
        return v.lower() if v else None


class PasswordChange(BaseModel):
    """修改密码请求模型"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator("new_password")
    def validate_new_password(cls, v):
        """验证新密码强度"""
        if len(v) < 8:
            raise ValueError("密码长度至少8个字符")
        if not any(c.isupper() for c in v):
            raise ValueError("密码必须包含至少一个大写字母")
        if not any(c.islower() for c in v):
            raise ValueError("密码必须包含至少一个小写字母")
        if not any(c.isdigit() for c in v):
            raise ValueError("密码必须包含至少一个数字")
        return v


class UserLogin(BaseModel):
    """用户登录请求模型"""
    username: str  # 可以是用户名或邮箱
    password: str
    remember_me: bool = False


class TokenResponse(BaseModel):
    """Token响应模型"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    user: UserResponse


class UserActivity(BaseModel):
    """用户活动记录"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    activity_type: str  # login, logout, chat_start, chat_end, etc.
    description: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
