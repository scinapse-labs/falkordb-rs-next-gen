from common import *
from neo4j import GraphDatabase
from neo4j.spatial import WGS84Point
import neo4j.graph
# from neo4j.debug import watch

bolt_con = None

class testBolt():
    def __init__(self):
        global bolt_con
        bolt_port = 7687
        self.env,_ = Env(moduleArgs=f"BOLT_PORT {bolt_port}")
        bolt_con = GraphDatabase.driver(f"bolt://localhost:{bolt_port}", auth=("falkordb", ""))
        # self.watcher = watch("neo4j")

    def test01_null(self):
        with bolt_con.session() as session:
            result = session.run("RETURN null, $v", {"v": None})
            record = result.single()
            self.env.assertEqual(record[0], None)

    def test02_boolean(self):
        with bolt_con.session() as session:
            result = session.run("RETURN true, false, $v_true, $v_false", {"v_true": True, "v_false": False})
            record = result.single()
            self.env.assertEqual(record[0], True)
            self.env.assertEqual(record[1], False)
            self.env.assertEqual(record[2], True)
            self.env.assertEqual(record[3], False)

    def test03_integer(self):
        with bolt_con.session() as session:
            result = session.run("RETURN -1, 0, 1, 2, $v", {"v": 3})
            record = result.single()
            self.env.assertEqual(record[0], -1)
            self.env.assertEqual(record[1], 0)
            self.env.assertEqual(record[2], 1)
            self.env.assertEqual(record[3], 2)
            self.env.assertEqual(record[4], 3)

            result = session.run("RETURN 255, 256, 257, $v", {"v": 258})
            record = result.single()
            self.env.assertEqual(record[0], 255)
            self.env.assertEqual(record[1], 256)
            self.env.assertEqual(record[2], 257)
            self.env.assertEqual(record[3], 258)

            result = session.run("RETURN 65535, 65536, 65537, $v", {"v": 65538})
            record = result.single()
            self.env.assertEqual(record[0], 65535)
            self.env.assertEqual(record[1], 65536)
            self.env.assertEqual(record[2], 65537)
            self.env.assertEqual(record[3], 65538)

            result = session.run("RETURN 4294967295, 4294967296, 4294967297, $v", {"v": 4294967298})
            record = result.single()
            self.env.assertEqual(record[0], 4294967295)
            self.env.assertEqual(record[1], 4294967296)
            self.env.assertEqual(record[2], 4294967297)
            self.env.assertEqual(record[3], 4294967298)

            result = session.run("RETURN 9223372036854775807, $v", {"v": 9223372036854775807})
            record = result.single()
            self.env.assertEqual(record[0], 9223372036854775807)
            self.env.assertEqual(record[1], 9223372036854775807)

    def test04_float(self):
        with bolt_con.session() as session:
            result = session.run("RETURN 1.23, $v", {"v": 4.56})
            record = result.single()
            self.env.assertEqual(record[0], 1.23)
            self.env.assertEqual(record[1], 4.56)

    def test05_string(self):
        with bolt_con.session() as session:
            result = session.run("RETURN '', 'Hello, World!', $v8, $v16", {"v8": 'A' * 255, "v16": 'A' * 256})
            record = result.single()
            self.env.assertEqual(record[0], '')
            self.env.assertEqual(record[1], 'Hello, World!')
            self.env.assertEqual(record[2], 'A' * 255)
            self.env.assertEqual(record[3], 'A' * 256)

    def test06_list(self):
        with bolt_con.session() as session:
            result = session.run("RETURN [], [1,2,3], $v8, $v16", {"v8": [1] * 255, "v16": [1] * 256})
            record = result.single()
            self.env.assertEqual(record[0], [])
            self.env.assertEqual(record[1], [1,2,3])
            self.env.assertEqual(record[2], [1] * 255)
            self.env.assertEqual(record[3], [1] * 256)

    def test07_map(self):
        with bolt_con.session() as session:
             result = session.run("RETURN {}, {foo:'bar'}, $v8", {"v8": {'foo':'bar'} })
             record = result.single()
             self.env.assertEqual(record[0], {})
             self.env.assertEqual(record[1], {'foo':'bar'})
             self.env.assertEqual(record[2], {'foo':'bar'})

    def test08_point(self):
         with bolt_con.session() as session:
             result = session.run("RETURN POINT({longitude:1, latitude:2})")
             record = result.single()
             self.env.assertEqual(record[0], WGS84Point((1, 2)))

    def test09_graph_entities_values(self):
         with bolt_con.session() as session:
             result = session.run("""CREATE (a:A {v: 1})-[r1:R1]->(b:B)<-[r2:R2]-(c:C) RETURN a, r1, b, r2, c""")
             record = result.single()
             a:neo4j.graph.Node = record[0]
             r1:neo4j.graph.Relationship = record[1]
             b:neo4j.graph.Node = record[2]
             r2:neo4j.graph.Relationship = record[3]
             c:neo4j.graph.Node = record[4]

             self.env.assertEqual(a.id, 0)
             self.env.assertEqual(a.labels, set(['A']))

             self.env.assertEqual(r1.id, 0)
             self.env.assertEqual(r1.type, 'R1')
             self.env.assertEqual(r1.start_node, a)
             self.env.assertEqual(r1.end_node, b)

             self.env.assertEqual(b.id, 1)
             self.env.assertEqual(b.labels, set(['B']))

             self.env.assertEqual(r2.id, 1)
             self.env.assertEqual(r2.type, 'R2')
             self.env.assertEqual(r2.start_node, c)
             self.env.assertEqual(r2.end_node, b)

             self.env.assertEqual(c.id, 2)
             self.env.assertEqual(c.labels, set(['C']))

             result = session.run("""MATCH p=(:A) RETURN p""")
             record = result.single()
             p:neo4j.graph.Path = record[0]
             self.env.assertEqual(p.start_node.labels, set(['A']))

             result = session.run("""MATCH p=(:A)-[:R1]->(:B) RETURN p""")
             record = result.single()
             p:neo4j.graph.Path = record[0]
             self.env.assertEqual(p.start_node.labels, set(['A']))
             self.env.assertEqual(p.end_node.labels, set(['B']))
             self.env.assertEqual(p.nodes[0].labels, set(['A']))
             self.env.assertEqual(p.nodes[1].labels, set(['B']))
             self.env.assertEqual(p.relationships[0].type, 'R1')

             result = session.run("""MATCH p=(:A)-[:R1]->(:B)<-[:R2]-(:C) RETURN p""")
             record = result.single()
             p:neo4j.graph.Path = record[0]
             self.env.assertEqual(p.start_node.labels, set(['A']))
             self.env.assertEqual(p.end_node.labels, set(['C']))
             self.env.assertEqual(p.nodes[0].labels, set(['A']))
             self.env.assertEqual(p.nodes[1].labels, set(['B']))
             self.env.assertEqual(p.nodes[2].labels, set(['C']))
             self.env.assertEqual(p.relationships[0].type, 'R1')
             self.env.assertEqual(p.relationships[1].type, 'R2')
