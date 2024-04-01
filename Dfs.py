class Dfs:
    
    def __init__(self,) -> None:
        self.frontier={}

    def add(self,node,pos):
        self.frontier[pos]=node
        return node
    
    def print_frontier(self):
        return self.frontier
    
    def empty(self):
        return len(self.frontier)==0

    def remove(self):
        if self.empty():
            raise Exception("Frontier is empty")
        else:
            
            #Save the last item in the list
            node=self.frontier[-1]
            #Save all the items in the list except the last
            self.frontier=self.frontier[:-1]
            return node
    
dfs_instance=Dfs()
print("The items in my frontier is: ", dfs_instance.print_frontier() )
print("The following item have been added",dfs_instance.add("A",3))
print("The following item have been added",dfs_instance.add("B",2))
print("The following item have been added",dfs_instance.add("C",1))
print("The following item have been added",dfs_instance.add("D",4))
print("The items in my frontier is: ", dfs_instance.print_frontier() )
print("The last item has been removed: ",dfs_instance.remove())
print("The items in my frontier is: ", dfs_instance.print_frontier() )

