from  model.KSOM.Node import *
class PNode(Node):
    def __init__(self,corpus, vector):
        super().__init__(vector)
        self.corpus = corpus
        
    def print_corpus(self):
        print(self.corpus)
    
    def __str__(self):
        return "CONTENT: "+self.corpus+"\n"