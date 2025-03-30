import pickle
import networkx as nx
import matplotlib.pyplot as plt


class QueryGraph(nx.DiGraph):
    def __init__(
        self,
        incoming_graph_data=None,
        query_base: str = None,
        pruning_thres: float = 0.5,
        **attr,
    ):
        super().__init__(incoming_graph_data, **attr)
        self.query_base = query_base
        self.pruning_thres = pruning_thres

    def update_cumulative_acc(self):
        for node in self.nodes():
            descendants = list(nx.descendants(self, node))
            if len(descendants) > 0:
                if self.query_base != "context":
                    attr = nx.get_node_attributes(self, "is_correct")
                    cum_acc = len([d for d in descendants if attr[d]]) / len(
                        descendants
                    )
                else:
                    attr = nx.get_node_attributes(self, "avg_acc")
                    cum_acc = sum([attr[d] for d in descendants]) / len(descendants)

                nx.set_node_attributes(self, {node: cum_acc}, "cumulative_acc")

    def pruning_by_cum_acc(self):
        """
        Find nodes with lowest cumulative accuracy.
        """
        selected_nodes = [
            node
            for node in self.nodes(data=True)
            if self.degree[node[0]] > 0
            and node[1]["cumulative_acc"] > self.pruning_thres
        ]
        for node in selected_nodes:
            self.remove_node(node[0])

    def pruning_by_avg_acc(self):
        """
        Find context with lowest average accuracy.
        """
        selected_nodes = [
            node
            for node in self.nodes(data=True)
            if node[1]["avg_acc"] > self.pruning_thres
        ]
        for node in selected_nodes:
            self.remove_node(node[0])

    def clean_graph(self):
        if self.query_base in ["statement", "question"]:
            selected_nodes = [
                node
                for node in self.nodes(data=True)
                if self.degree[node[0]] == 0 and node[1]["is_correct"]
            ]
        else:
            selected_nodes = [
                node
                for node in self.nodes(data=True)
                if self.degree[node[0]] == 0 and node[1]["avg_acc"] > self.pruning_thres
            ]

        for node in selected_nodes:
            self.remove_node(node[0])

        # remove loop
        selected_edges = [
            (edge[0], edge[1]) for edge in self.edges() if edge[0] == edge[1]
        ]
        for n0, n1 in selected_edges:
            self.remove_edge(n0, n1)

    def show(self, fig_save_path: str, step: int):
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.set_title("Q-Graph", fontsize=10)
        pos = nx.nx_agraph.graphviz_layout(self)
        if self.query_base in ["statement", "question"]:
            node_colors = [
                "green" if self.nodes[n]["is_correct"] else "red" for n in self.nodes()
            ]
        else:
            node_colors = [
                "green" if self.nodes[n]["avg_acc"] > self.pruning_thres else "red"
                for n in self.nodes()
            ]
        nx.draw(
            self,
            pos,
            node_size=150,
            node_color=node_colors,
            font_size=8,
            font_weight="bold",
        )
        plt.tight_layout()
        plt.savefig(f"{fig_save_path}/{step}.jpg", bbox_inches="tight")
        plt.close()

    def save(self, save_path: str, step: int):
        pickle.dump(self, open(f"{save_path}/{step}_qg.pkl", "wb"))

    def load(self, path: str, step: int):
        return pickle.load(open(f"{path}/{str(step)}_qg.pkl", "rb"))