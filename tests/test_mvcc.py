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
    return (id, res._raw_stats)


def run_read(id):
    db = FalkorDB()
    g = db.select_graph("test")
    res = g.query("MATCH (n:Node) RETURN n.id")
    return (id, res.result_set, res._raw_stats)


def test_mvcc():
    common.g.query("UNWIND range(1, 1000) AS x CREATE (:Node {id: x})")

    pool1 = Pool(1)
    pool8 = Pool(8)

    res_write = pool1.map_async(run_write, range(1, 100))
    res_read = pool8.map_async(run_read, range(1, 100))

    res_write.wait()
    res_read.wait()

    # res = common.g.query("MATCH (n:Node) RETURN n.id")
    # print(res.result_set)
    
    # for r in res_write.get():
    #     print(r[0], r[1][4])

    for r in res_read.get():
        version = int(r[2][2][15:])
        res = r[1]
        # print(r[0], version, res[0:version])
        assert res[0:version - 1] == [[0] for i in range(1, version)]
        assert res[version - 1] == [version]
