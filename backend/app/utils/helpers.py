"""
辅助工具函数

提供各种通用的辅助功能
"""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID


def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化日期时间"""
    return dt.strftime(format_str)


def generate_uuid() -> UUID:
    """生成UUID"""
    return uuid.uuid4()


def validate_email(email: str) -> bool:
    """验证邮箱格式"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除不安全字符"""
    # 移除或替换不安全字符
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除前后空格和点
    sanitized = sanitized.strip('. ')
    # 限制长度
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized or "untitled"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """提取关键词（简单实现）"""
    # 移除标点和转换为小写
    clean_text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text.lower())
    
    # 分词
    words = clean_text.split()
    
    # 过滤停用词（简化版）
    stop_words = {
        '的', '了', '在', '是', '我', '你', '他', '她', '它', '们',
        '这', '那', '有', '和', '与', '或', '但', '如果', '因为',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
    }
    
    # 过滤停用词和短词
    keywords = [word for word in words 
               if len(word) > 2 and word not in stop_words]
    
    # 计算词频
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序并返回
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:max_keywords]]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """计算文本相似度（简单实现）"""
    # 提取关键词
    keywords1 = set(extract_keywords(text1, 20))
    keywords2 = set(extract_keywords(text2, 20))
    
    if not keywords1 and not keywords2:
        return 1.0
    if not keywords1 or not keywords2:
        return 0.0
    
    # 计算Jaccard相似度
    intersection = keywords1 & keywords2
    union = keywords1 | keywords2
    
    return len(intersection) / len(union)


def parse_duration(duration_str: str) -> Optional[int]:
    """解析时间长度字符串，返回秒数"""
    pattern = r'(\d+)\s*(s|sec|second|seconds|m|min|minute|minutes|h|hour|hours|d|day|days)'
    match = re.match(pattern, duration_str.lower().strip())
    
    if not match:
        return None
    
    value, unit = match.groups()
    value = int(value)
    
    multipliers = {
        's': 1, 'sec': 1, 'second': 1, 'seconds': 1,
        'm': 60, 'min': 60, 'minute': 60, 'minutes': 60,
        'h': 3600, 'hour': 3600, 'hours': 3600,
        'd': 86400, 'day': 86400, 'days': 86400
    }
    
    return value * multipliers.get(unit, 1)


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def deep_merge_dict(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并字典"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dict(result[key], value)
        else:
            result[key] = value
    
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """展平嵌套字典"""
    items = []
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def batch_process(items: List[Any], batch_size: int = 100):
    """批处理生成器"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        import functools
        import asyncio
        import time
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # 指数退避
                    
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # 指数退避
                    
            raise last_exception
        
        # 检查函数是否是协程
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """安全的JSON解析"""
    try:
        import json
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """掩码敏感数据"""
    if len(data) <= visible_chars * 2:
        return mask_char * len(data)
    
    return data[:visible_chars] + mask_char * (len(data) - visible_chars * 2) + data[-visible_chars:]


class RateLimiter:
    """简单的速率限制器"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = {}
    
    def is_allowed(self, key: str) -> bool:
        """检查是否允许调用"""
        import time
        
        current_time = time.time()
        
        # 清理过期记录
        if key in self.calls:
            self.calls[key] = [
                call_time for call_time in self.calls[key]
                if current_time - call_time < self.time_window
            ]
        else:
            self.calls[key] = []
        
        # 检查是否超过限制
        if len(self.calls[key]) >= self.max_calls:
            return False
        
        # 记录本次调用
        self.calls[key].append(current_time)
        return True


def generate_short_id(length: int = 8) -> str:
    """生成短ID"""
    import string
    import random
    
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def is_valid_uuid(uuid_string: str) -> bool:
    """验证UUID格式"""
    try:
        UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False
