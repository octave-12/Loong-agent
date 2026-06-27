#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
裂变架构 (Fission Architecture) — 个体裂变与知识融合

设计原则:
  - 核心知识共享 (concept_graph.db): 所有实例共享同一知识库
  - 经验隔离: 每个实例有自己的对话历史、用户偏好、学习轨迹
  - 定期融合: 实例验证通过的知识写回共享层

架构:
  InstanceIdentity  — 个体身份标识 (UUID, role, capabilities)
  SharedKnowledge    — 共享知识层 (读写 concept_graph.db)
  LocalExperience    — 个体私有经验 (不共享, JSON)
  FissionManager     — 裂变/融合/同步编排器

用法:
    from loongpearl.core.fission import FissionManager

    fm = FissionManager()
    seed = fm.spawn(role='seed')
    worker = fm.spawn(role='worker')
    # worker 学习...
    fm.fuse(worker.instance_id, learned_triples)
    fm.sync_instance(seed.instance_id)
"""

import os
import json
import uuid
import time
import sqlite3
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

log = logging.getLogger(__name__)


# ============================================================================
# 1. InstanceIdentity — 个体身份标识
# ============================================================================

@dataclass
class InstanceIdentity:
    """个体实例身份标识

    Attributes:
        instance_id:   UUID 唯一标识
        created_at:    创建时间 (ISO 8601)
        last_sync_at:  最后同步共享层时间
        role:          角色: 'seed'|'worker'|'explorer'|'teacher'
        capabilities:  能力列表 ['knowledge_retrieval','learning','dialogue',...]
        last_pulled_id: 共享知识库上次同步到的最大 rowid (增量同步锚点)
    """
    instance_id: str
    created_at: str
    last_sync_at: str
    role: str
    capabilities: List[str] = field(default_factory=list)
    last_pulled_id: int = 0

    # ── 角色默认能力表 ──
    DEFAULT_CAPABILITIES = {
        'seed':     ['knowledge_retrieval', 'learning', 'dialogue', 'spawn', 'teach'],
        'worker':   ['knowledge_retrieval', 'learning', 'dialogue'],
        'explorer': ['knowledge_retrieval', 'learning', 'search', 'extract', 'hypothesis_test'],
        'teacher':  ['knowledge_retrieval', 'dialogue', 'teach', 'verify', 'curate'],
    }

    @classmethod
    def create(cls, role: str = 'seed') -> 'InstanceIdentity':
        """工厂方法: 创建新实例身份"""
        now = datetime.now(timezone.utc).isoformat()
        capabilities = list(cls.DEFAULT_CAPABILITIES.get(
            role, cls.DEFAULT_CAPABILITIES['worker']
        ))
        return cls(
            instance_id=str(uuid.uuid4()),
            created_at=now,
            last_sync_at=now,
            role=role,
            capabilities=capabilities,
        )

    def save(self, data_dir: str = 'data/instances') -> str:
        """持久化到 data/instances/{instance_id}/identity.json"""
        instance_dir = os.path.join(data_dir, self.instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        path = os.path.join(instance_dir, 'identity.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, instance_id: str,
             data_dir: str = 'data/instances') -> Optional['InstanceIdentity']:
        """从 JSON 加载实例身份"""
        path = os.path.join(data_dir, instance_id, 'identity.json')
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def touch_sync(self):
        """更新最后同步时间"""
        self.last_sync_at = datetime.now(timezone.utc).isoformat()

    def __repr__(self):
        return (f"InstanceIdentity(id={self.instance_id[:8]}..., "
                f"role={self.role}, caps={len(self.capabilities)})")


# ============================================================================
# 2. SharedKnowledge — 共享知识层
# ============================================================================

class SharedKnowledge:
    """共享知识层 — 所有实例共享的 concept_graph.db

    封装对 SQLite 概念图的读写，提供:
      - pull_since(last_id):  增量拉取新三元组
      - push_triples(triples): 写入验证通过的三元组（带来源实例ID）
      - get_instance_contributions(): 各实例贡献统计

    兼容现有 concept_graph.db 的 triples 表结构:
      id, s, r, o, c, src, ev, learned_at, last_verified_at, verify_count
    新增列:
      instance_id  — 贡献实例 UUID
      created_at   — 三元组创建时间戳
    """

    def __init__(self, db_path: str = 'data/models/concept_graph.db'):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self):
        """确保裂变所需的表和列存在（兼容现有数据库）"""
        c = self.conn

        # 1. 创建 triples 表（如果不存在）
        c.execute("""
            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                s TEXT NOT NULL,
                r TEXT NOT NULL,
                o TEXT NOT NULL,
                c REAL DEFAULT 1.0,
                src TEXT DEFAULT '',
                ev TEXT DEFAULT '',
                learned_at TEXT DEFAULT '',
                last_verified_at TEXT DEFAULT '',
                verify_count INTEGER DEFAULT 0
            )
        """)
        # 创建基础索引（如果不存在）
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_triples_s ON triples(s)",
            "CREATE INDEX IF NOT EXISTS idx_triples_r ON triples(r)",
            "CREATE INDEX IF NOT EXISTS idx_triples_o ON triples(o)",
            "CREATE INDEX IF NOT EXISTS idx_triples_sr ON triples(s, r)",
        ]:
            c.execute(idx_sql)

        # 2. 确保裂变专用列存在
        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(triples)")}
        migrations = [
            ('instance_id', "ALTER TABLE triples ADD COLUMN instance_id TEXT DEFAULT ''"),
            ('created_at',   "ALTER TABLE triples ADD COLUMN created_at TEXT DEFAULT ''"),
        ]
        for col, sql in migrations:
            if col not in existing_cols:
                try:
                    c.execute(sql)
                    log.info(f"SharedKnowledge: 添加列 triples.{col}")
                except sqlite3.OperationalError as e:
                    log.warning(f"添加列 {col} 失败: {e}")
        c.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ═══════════════════════════════════════════════════════════════════════
    # 读取
    # ═══════════════════════════════════════════════════════════════════════

    def pull_since(self, last_pulled_id: int = 0) -> List[Dict]:
        """增量拉取: 获取 id > last_pulled_id 的新增三元组

        Args:
            last_pulled_id: 上次同步到的最大 rowid

        Returns:
            [{id, s, r, o, c, src, ev, instance_id, created_at}, ...]
        """
        rows = self.conn.execute(
            "SELECT id, s, r, o, c, src, ev, instance_id, created_at "
            "FROM triples WHERE id > ? ORDER BY id ASC",
            (last_pulled_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def pull_since_timestamp(self, since: str) -> List[Dict]:
        """按时间戳拉取: 获取 created_at > since 的三元组

        Args:
            since: ISO 时间戳字符串

        Returns:
            [{id, s, r, o, c, src, ev, instance_id, created_at}, ...]
        """
        rows = self.conn.execute(
            "SELECT id, s, r, o, c, src, ev, instance_id, created_at "
            "FROM triples WHERE created_at > ? ORDER BY id ASC",
            (since,)
        ).fetchall()
        return [dict(row) for row in rows]

    def get_max_id(self) -> int:
        """获取当前最大 rowid"""
        row = self.conn.execute("SELECT MAX(id) FROM triples").fetchone()
        return row[0] if row[0] else 0

    def get_triple_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]

    # ═══════════════════════════════════════════════════════════════════════
    # 写入
    # ═══════════════════════════════════════════════════════════════════════

    def push_triples(self, triples: List[Dict], instance_id: str,
                     validate: bool = True) -> Tuple[int, int]:
        """写入验证通过的三元组到共享层

        验证规则:
          - s, r, o 非空
          - s, o 长度 ≤ 12 (过滤长文本)
          - s ≠ o (不自指)
          - confidence 在 [0.0, 1.0]

        去重策略: 已存在的 (s,r,o) 更新置信度（取最大值）并增加验证计数

        Args:
            triples: [{'s':..., 'r':..., 'o':..., 'c':..., 'src':..., 'ev':...}, ...]
            instance_id: 贡献实例 UUID
            validate: 是否执行验证

        Returns:
            (accepted, rejected)
        """
        now = datetime.now(timezone.utc).isoformat()
        accepted = 0
        rejected = 0

        for t in triples:
            if validate and not self._validate_triple(t):
                rejected += 1
                continue

            existing = self.conn.execute(
                "SELECT id, c FROM triples WHERE s=? AND r=? AND o=? LIMIT 1",
                (t['s'], t['r'], t['o'])
            ).fetchone()

            if existing:
                # 已存在: 置信度取最大值，增加验证计数
                new_c = max(existing['c'], t.get('c', 0.5))
                self.conn.execute(
                    "UPDATE triples SET c=?, instance_id=?, "
                    "last_verified_at=?, verify_count=verify_count+1 "
                    "WHERE id=?",
                    (new_c, instance_id, now, existing['id'])
                )
            else:
                # 新三元组
                self.conn.execute(
                    "INSERT INTO triples(s, r, o, c, src, ev, instance_id, "
                    "created_at, learned_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        t['s'], t['r'], t['o'],
                        t.get('c', 0.5),
                        t.get('src', 'fission'),
                        t.get('ev', ''),
                        instance_id,
                        now,
                        now,
                    )
                )
            accepted += 1

        self.conn.commit()
        return accepted, rejected

    @staticmethod
    def _validate_triple(t: Dict) -> bool:
        """验证三元组基本有效性"""
        s = t.get('s', '')
        r = t.get('r', '')
        o = t.get('o', '')
        if not s or not r or not o:
            return False
        if len(s) > 12 or len(o) > 12:
            return False
        if s == o:
            return False
        c = t.get('c', 0.5)
        if not (0.0 <= float(c) <= 1.0):
            return False
        return True

    # ═══════════════════════════════════════════════════════════════════════
    # 统计
    # ═══════════════════════════════════════════════════════════════════════

    def get_instance_contributions(self) -> Dict[str, int]:
        """各实例贡献统计 → {instance_id: triple_count}"""
        rows = self.conn.execute(
            "SELECT instance_id, COUNT(*) as cnt FROM triples "
            "WHERE instance_id != '' GROUP BY instance_id ORDER BY cnt DESC"
        ).fetchall()
        return {row['instance_id']: row['cnt'] for row in rows}

    def stats(self) -> Dict:
        """综合统计"""
        return {
            'total_triples': self.get_triple_count(),
            'instance_contributions': self.get_instance_contributions(),
        }


# ============================================================================
# 3. LocalExperience — 个体私有经验
# ============================================================================

@dataclass
class LocalExperience:
    """个体私有数据 — 每个实例独立，不共享

    存储内容:
      - dialogue_history:   对话历史 [{role, content, timestamp}, ...]
      - user_preferences:   用户偏好 {key: value}
      - learning_trajectory: 学习轨迹 [{type, description, metadata, timestamp}, ...]
      - custom_data:         扩展数据

    持久化路径: data/instances/{instance_id}/experience.json
    """

    instance_id: str
    dialogue_history: List[Dict] = field(default_factory=list)
    user_preferences: Dict = field(default_factory=dict)
    learning_trajectory: List[Dict] = field(default_factory=list)
    custom_data: Dict = field(default_factory=dict)

    # ── 写入方法 ──

    def add_dialogue(self, role: str, content: str) -> Dict:
        """添加一条对话记录"""
        entry = {
            'role': role,
            'content': content,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        self.dialogue_history.append(entry)
        return entry

    def add_learning_event(self, event_type: str, description: str,
                           metadata: Dict = None) -> Dict:
        """添加一条学习轨迹事件"""
        entry = {
            'type': event_type,
            'description': description,
            'metadata': metadata or {},
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        self.learning_trajectory.append(entry)
        return entry

    def set_preference(self, key: str, value):
        """设置用户偏好"""
        self.user_preferences[key] = value

    # ── 持久化 ──

    def save(self, data_dir: str = 'data/instances') -> str:
        """持久化到 JSON 文件"""
        instance_dir = os.path.join(data_dir, self.instance_id)
        os.makedirs(instance_dir, exist_ok=True)
        path = os.path.join(instance_dir, 'experience.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, instance_id: str,
             data_dir: str = 'data/instances') -> 'LocalExperience':
        """加载私有经验，不存在则返回空实例"""
        path = os.path.join(data_dir, instance_id, 'experience.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls(**data)
        return cls(instance_id=instance_id)

    def __repr__(self):
        return (f"LocalExperience(id={self.instance_id[:8]}..., "
                f"dialogues={len(self.dialogue_history)}, "
                f"learn_events={len(self.learning_trajectory)})")


# ============================================================================
# 4. FissionManager — 裂变/融合/同步编排器
# ============================================================================

class FissionManager:
    """裂变管理器 — 编排个体完整生命周期

    核心操作:
      spawn(role)       → 从当前生态裂变出新实例
      fuse(instance_id) → 将该实例验证通过的知识合并到共享层
      sync_all()        → 所有实例同步共享层最新知识
      list_instances()  → 列出所有存活实例
    """

    def __init__(self, db_path: str = 'data/models/concept_graph.db',
                 data_dir: str = 'data/instances'):
        self.db_path = db_path
        self.data_dir = data_dir
        self.shared = SharedKnowledge(db_path)
        os.makedirs(data_dir, exist_ok=True)
        log.info(f"FissionManager 初始化: db={db_path}, instances={data_dir}")

    # ═══════════════════════════════════════════════════════════════════════
    # 裂变
    # ═══════════════════════════════════════════════════════════════════════

    def spawn(self, role: str = 'worker',
              parent_id: str = None) -> InstanceIdentity:
        """从当前实例裂变出新实例

        裂变过程:
          1. 创建新身份 (UUID, role, capabilities)
          2. 锚定共享知识层当前 max_id (后续增量同步)
          3. 初始化空白私有经验
          4. 持久化身份和经验

        Args:
            role: 'seed'|'worker'|'explorer'|'teacher'
            parent_id: 父实例 ID (可选)

        Returns:
            新实例的 InstanceIdentity
        """
        # 1. 创建身份
        identity = InstanceIdentity.create(role)
        identity.last_pulled_id = self.shared.get_max_id()

        # 2. 初始化私有经验
        experience = LocalExperience(instance_id=identity.instance_id)

        # 记录裂变事件
        if parent_id:
            experience.add_learning_event(
                'spawn',
                f'从 {parent_id[:8]}... 裂变而来',
                {'parent_id': parent_id, 'role': role}
            )
        else:
            experience.add_learning_event(
                'genesis',
                f'初始 {role} 实例诞生',
                {'role': role}
            )

        # 3. 持久化
        identity.save(self.data_dir)
        experience.save(self.data_dir)

        log.info(f"🐉 裂变: {identity}")
        return identity

    # ═══════════════════════════════════════════════════════════════════════
    # 融合
    # ═══════════════════════════════════════════════════════════════════════

    def fuse(self, instance_id: str,
             triples: List[Dict] = None) -> Tuple[int, int]:
        """将该实例验证通过的知识合并到共享层

        Args:
            instance_id: 实例 UUID
            triples: 要融合的三元组列表；为 None 则从实例经验提取

        Returns:
            (accepted_count, rejected_count)
        """
        identity = InstanceIdentity.load(instance_id, self.data_dir)
        if identity is None:
            log.warning(f"融合失败: 实例 {instance_id[:8]}... 不存在")
            return (0, 0)

        if triples is None:
            triples = []

        accepted, rejected = self.shared.push_triples(triples, instance_id)

        # 更新同步锚点
        identity.last_pulled_id = self.shared.get_max_id()
        identity.touch_sync()
        identity.save(self.data_dir)

        # 记录融合事件
        experience = LocalExperience.load(instance_id, self.data_dir)
        experience.add_learning_event(
            'fuse',
            f'融合 {accepted} 条知识到共享层 (拒绝 {rejected})',
            {'accepted': accepted, 'rejected': rejected}
        )
        experience.save(self.data_dir)

        log.info(f"🔄 融合: {instance_id[:8]}... accepted={accepted} rejected={rejected}")
        return (accepted, rejected)

    # ═══════════════════════════════════════════════════════════════════════
    # 列出
    # ═══════════════════════════════════════════════════════════════════════

    def list_instances(self) -> List[InstanceIdentity]:
        """列出所有存活实例"""
        instances = []
        if not os.path.isdir(self.data_dir):
            return instances

        for dirname in sorted(os.listdir(self.data_dir)):
            identity = InstanceIdentity.load(dirname, self.data_dir)
            if identity:
                instances.append(identity)

        return instances

    def get_instance(self, instance_id: str) -> Optional[InstanceIdentity]:
        return InstanceIdentity.load(instance_id, self.data_dir)

    # ═══════════════════════════════════════════════════════════════════════
    # 同步
    # ═══════════════════════════════════════════════════════════════════════

    def sync_all(self) -> Dict[str, int]:
        """所有实例同步共享层最新知识

        Returns:
            {instance_id: new_triple_count}
        """
        results = {}
        for identity in self.list_instances():
            new_triples = self.shared.pull_since(identity.last_pulled_id)
            identity.last_pulled_id = self.shared.get_max_id()
            identity.touch_sync()
            identity.save(self.data_dir)
            results[identity.instance_id] = len(new_triples)

        log.info(f"📡 同步完成: {len(results)} 个实例, "
                 f"总计 {sum(results.values())} 条新知识")
        return results

    def sync_instance(self, instance_id: str) -> int:
        """同步单个实例

        Returns:
            新增三元组数量
        """
        identity = InstanceIdentity.load(instance_id, self.data_dir)
        if identity is None:
            log.warning(f"同步失败: 实例 {instance_id[:8]}... 不存在")
            return 0

        new_triples = self.shared.pull_since(identity.last_pulled_id)
        identity.last_pulled_id = self.shared.get_max_id()
        identity.touch_sync()
        identity.save(self.data_dir)

        if new_triples:
            log.info(f"📡 同步: {instance_id[:8]}... +{len(new_triples)} 条")
        return len(new_triples)

    # ═══════════════════════════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════════════════════════

    def close(self):
        self.shared.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# 5. __main__ 测试
# ============================================================================

if __name__ == '__main__':
    import tempfile

    print("=" * 60)
    print("🐉 龙珠裂变架构 — 测试演示")
    print("=" * 60)

    # 使用临时目录测试，避免污染真实数据
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'concept_graph.db')
        data_dir = os.path.join(tmpdir, 'instances')

        print(f"\n📁 临时测试目录: {tmpdir}")

        fm = FissionManager(db_path=db_path, data_dir=data_dir)

        # ── 1. 创建 seed 实例 ──
        print("\n" + "─" * 40)
        print("1️⃣  创建 seed 实例...")
        seed = fm.spawn(role='seed')
        print(f"   ID:       {seed.instance_id[:8]}...")
        print(f"   Role:     {seed.role}")
        print(f"   Caps:     {seed.capabilities}")
        print(f"   Created:  {seed.created_at[:19]}")

        # ── 2. spawn 一个 worker ──
        print("\n" + "─" * 40)
        print("2️⃣  裂变 worker 实例...")
        worker = fm.spawn(role='worker', parent_id=seed.instance_id)
        print(f"   ID:       {worker.instance_id[:8]}...")
        print(f"   Role:     {worker.role}")
        print(f"   Parent:   {seed.instance_id[:8]}...")

        # spawn 一个 explorer
        explorer = fm.spawn(role='explorer', parent_id=seed.instance_id)
        print(f"   Explorer: {explorer.instance_id[:8]}... (role={explorer.role})")

        # ── 3. worker 学点知识, fuse 回共享层 ──
        print("\n" + "─" * 40)
        print("3️⃣  Worker 学习知识并融合...")
        worker_triples = [
            {'s': '电子', 'r': 'PART_OF', 'o': '原子', 'c': 0.90, 'src': 'worker_learn'},
            {'s': '原子', 'r': 'PART_OF', 'o': '分子', 'c': 0.85, 'src': 'worker_learn'},
            {'s': '分子', 'r': 'PART_OF', 'o': '细胞', 'c': 0.80, 'src': 'worker_learn'},
            # 这条无效 (s==o) 应被拒绝
            {'s': '测试', 'r': 'IS_A', 'o': '测试', 'c': 0.5, 'src': 'worker_learn'},
            # 这条有效
            {'s': '夸克', 'r': 'PART_OF', 'o': '质子', 'c': 0.75, 'src': 'worker_learn'},
        ]
        accepted, rejected = fm.fuse(worker.instance_id, worker_triples)
        print(f"   融合结果: accepted={accepted}, rejected={rejected}")

        # ── 4. seed pull 最新知识 ──
        print("\n" + "─" * 40)
        print("4️⃣  Seed 同步最新知识...")
        new_count = fm.sync_instance(seed.instance_id)
        print(f"   Seed 同步到 {new_count} 条新知识")

        # ── 5. 验证共享知识库 ──
        print("\n" + "─" * 40)
        print("5️⃣  共享知识库内容:")
        all_triples = fm.shared.pull_since(0)
        print(f"   总计 {len(all_triples)} 条三元组:")
        for t in all_triples:
            print(f"     {t['s']:6s} --{t['r']:9s}--> {t['o']:4s}  "
                  f"(c={t['c']:.2f}, by={t['instance_id'][:8] if t['instance_id'] else 'N/A'})")

        # ── 6. 列表示例 ──
        print("\n" + "─" * 40)
        print("6️⃣  所有实例及经验:")
        for inst in fm.list_instances():
            exp = LocalExperience.load(inst.instance_id, data_dir)
            print(f"   [{inst.role:8s}] {inst.instance_id[:8]}...  "
                  f"sync_pull_id={inst.last_pulled_id:2d}  "
                  f"events={len(exp.learning_trajectory):2d}  "
                  f"dialogues={len(exp.dialogue_history)}")

        # ── 7. 贡献统计 ──
        print("\n" + "─" * 40)
        print("7️⃣  实例贡献统计:")
        contributions = fm.shared.get_instance_contributions()
        for inst_id, count in contributions.items():
            print(f"   {inst_id[:8]}...: {count} 条三元组")
        if not contributions:
            print("   (无贡献)")

        # ── 8. Explorer 也学点并融合 ──
        print("\n" + "─" * 40)
        print("8️⃣  Explorer 学习并融合...")
        explorer_triples = [
            {'s': '质子', 'r': 'PART_OF', 'o': '原子核', 'c': 0.88, 'src': 'explorer'},
            {'s': '中子', 'r': 'PART_OF', 'o': '原子核', 'c': 0.88, 'src': 'explorer'},
        ]
        fm.fuse(explorer.instance_id, explorer_triples)

        # ── 9. 全量同步 ──
        print("\n" + "─" * 40)
        print("9️⃣  全量同步 sync_all()...")
        sync_results = fm.sync_all()
        for iid, count in sync_results.items():
            print(f"   {iid[:8]}...: +{count} 条新知识")

        # ── 10. 最终统计 ──
        print("\n" + "─" * 40)
        print("🔟 最终统计:")
        stats = fm.shared.stats()
        print(f"   三元组总数: {stats['total_triples']}")
        print(f"   存活实例数: {len(fm.list_instances())}")
        contribs = stats['instance_contributions']
        for iid, cnt in contribs.items():
            inst = fm.get_instance(iid)
            role = inst.role if inst else '?'
            print(f"   [{role}] {iid[:8]}...: {cnt} 条")

        fm.close()

    print("\n" + "=" * 60)
    print("✅ 裂变架构测试完成!")
    print("=" * 60)
