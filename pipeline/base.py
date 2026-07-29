# Its purpose is just to increase the reusability in pipelines
class Base:
    def __init__(self,dataset,target_col):
        self.dataset=dataset
        self.target_col=target_col

