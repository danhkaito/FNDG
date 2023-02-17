from  model.KSOM.Node import *
class PNode(Node):
    def __init__(self,corpus, label, vector):
        super().__init__(vector)
        self.corpus = corpus
        self.label = label
        
    def print_corpus(self):
        print(self.corpus)
        
    def print_label(self):
        print(self.label)