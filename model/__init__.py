"""正式 AMS-SemiCD 模型导出。

项目运行入口统一使用 ``SemiModel``（论文中的 ``SemiModel_ALL``）。
"""

from .model import SemiModel

__all__ = ["SemiModel"]
