from common import *

GRAPH_KEY = "execution_plan_print"

class test_execution_plan_print():
    def __init__(self):
        self.env, self.db = Env()
        self.graph = self.db.select_graph(GRAPH_KEY)

        # create key
        self.graph.query("RETURN 1")
    
    # label scan handles all labels for the node
    def test01_multi_label_scan(self):
        plan = str(self.graph.explain("MATCH (n:A:B) RETURN n"))

        # all labels are handled by the label scan
        self.env.assertContains("Node By Label Scan | (n:A:B)", plan)

        # no separate conditional traverse for the extra label
        self.env.assertNotContains("Conditional Traverse", plan)

    # label scan handles all labels for the node
    def test02_multi_label_scan(self):
        plan = str(self.graph.explain("MATCH (n:A:B:C) RETURN n"))

        # all labels are handled by the label scan
        self.env.assertContains("Node By Label Scan | (n:A:B:C)", plan)

        # no separate expand into for the extra labels
        self.env.assertNotContains("Expand Into", plan)

    # variable length traverse with labeled endpoints
    def test03_variable_length_traverse(self):
        plan = str(self.graph.explain("match p=(n:A:B)-[*]-(m:C:D) RETURN p"))

        # variable length traverse handles the traversal
        self.env.assertContains("Variable Length Traverse | (n)-[_anon_0]-(m)", plan)

