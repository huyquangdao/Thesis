import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from scripts.models import constants
from scripts.dataset.module import plotlib


class DepTree:
    def __init__(self, edges=None, tree=None):
        """
        :param list of (str, models.Token, models.Token) edges: relation from token to token
        """
        if tree:
            # construct from networkx graph
            self.tree = tree
        else:
            # construct from edges
            if edges is None:
                raise ValueError('tree or edges must be nonNone')

            self.tree = nx.DiGraph()

            for e in edges:

                self.tree.add_node(e[1].get_node(), pos_tag=e[1].metadata['pos_tag'], label=e[1].content, token=e[1])
                self.tree.add_node(e[2].get_node(), pos_tag=e[2].metadata['pos_tag'], label=e[2].content, token=e[2])
                self.tree.add_edge(e[1].get_node(), e[2].get_node(), relation=e[0], type='dependency', weight=1.0)

        # print('edge', edges)

        # validate tree
        if not nx.is_tree(self.tree):
            raise ValueError('Invalid tree')

        # get root
        self.root = None
        for node in self.tree.nodes():
            if not self.tree.predecessors(node):
                if self.root is None:
                    self.root = node
                else:
                    raise ValueError('Double root')
        to_nodes = []
        for edge in self.tree.edges():
            to_nodes.append(edge[1])

        root = []

        for node in self.tree.nodes():
            if node not in to_nodes:
                for _,e1,_ in edges:
                    if e1.get_node() == node:
                        root.append(e1)
                        break

        if len(root) > 1 or len(root) == 0:
            raise ValueError('Invalid Root')

        self.root = root[0]

    def clean_leaf(self):
        """
        :return:
        """
        for node, node_data in list(self.tree.nodes(data=True)):
            if node_data['token'].is_stop_word() and self.is_leaf(node):
                self.tree.remove_node(node)

    def is_leaf(self, node):
        """

        :param node:
        :return:
        """
        if len(self.tree.successors(node)) != 0:
            return False
        else:
            return True

    def get_sub_tree(self, node):
        """
        :param (str, (int, int), (int, int)) node: (content, doc_offset, sent_offset)
        :return: DepTree
        """
        dfs = nx.dfs_tree(self.tree, node)
        subtree = self.tree.subgraph(dfs.nodes())
        for node in subtree.nodes():
            subtree.node[node].update(self.tree.nodes()[node])
        return DepTree(tree=subtree)

    def find_nearest_verb(self, node):
        parents = self.tree.predecessors(node)
        if not parents:
            return node
        else:
            father = parents[0]
            pos_tag = self.tree.nodes()[father]['pos_tag']
            if pos_tag in constants.VERB_TAGS:
                return father
            else:
                return self.find_nearest_verb(father)

    def find_root_tree(self, node):
        path = nx.shortest_path(self.tree, self.root, node)
        subtree = self.tree.subgraph(path)
        for node in subtree.nodes():
            subtree.node[node].update(self.tree.nodes()[node])

        return DepTree(tree=subtree)

    def get_root(self, data=False):
        if data:
            return self.root, self.tree.nodes()[self.root]
        else:
            return self.root

    def visualize(self):
        plotlib.next_plot()
        node_labels = nx.get_node_attributes(self.tree, 'label')
        nx.draw_networkx(
            self.tree,
            labels=node_labels,
            pos=graphviz_layout(self.tree, prog='dot'),
            node_size=300,
            font_weight='bold',
            font_color='xkcd:red',
            node_color='xkcd:light grey'
        )
