# 工具函数
import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    # 解析 "module@Class" 并动态导入
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)
    
    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    # 返回模块源码路径（用于保存代码快照）
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)
