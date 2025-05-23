import os, sys
# 插入 rl/ 目录到 sys.path
sys.path.insert(0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)
