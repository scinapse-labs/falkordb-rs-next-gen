import common
from falkordb import FalkorDB
from multiprocessing import Pool


def setup_module(module):
    global is_extra
    from conftest import pytest_config

    is_extra = "extra" in pytest_config.getoption("-m")
    common.start_redis()


def teardown_module(module):
    common.shutdown_redis()


def setup_function(function):
    if common.g.name in common.client.list_graphs():
        common.g.delete()


def run_write(id):
    db = FalkorDB()
    g = db.select_graph("test")
    res = g.query("MATCH (n:Node) WHERE n.id = $id SET n.id = 0", params={"id": id})
    version = int(res._raw_stats[4][15:])
    return (id, version)


def run_read(id):
    db = FalkorDB()
    g = db.select_graph("test")
    res = g.query("MATCH (n:Node) RETURN n.id")
    version = int(res._raw_stats[2][15:])
    return (id, res.result_set, version)


def test_mvcc():
    common.g.query("UNWIND range(1, 1000) AS x CREATE (:Node {id: x})")
    try:
        common.g.query("UNWIND [1, 2, 3] AS x MATCH (n:Node {id: x}) SET n.id = 1 / (x - 3)")
    except:
        pass

    res = common.g.query("MATCH (n:Node) RETURN n.id")
    assert res.result_set == [[x] for x in range(1, 1001)]

    pool1 = Pool(1)
    pool8 = Pool(8)

    res_write = pool1.map_async(run_write, range(1, 100))
    res_read = pool8.map_async(run_read, range(1, 100))

    res_write.wait()
    res_read.wait()

    res = common.g.query("MATCH (n:Node) RETURN n.id")
    assert res.result_set == [[x if x >= 100 else 0] for x in range(1, 1001)]
    
    assert res_write.get() == [(x, x + 1) for x in range(1, 100)]

    for r in res_read.get():
        version = r[2]
        res = r[1]
        assert res == [[0 if i < version else i] for i in range(1, 1001)]