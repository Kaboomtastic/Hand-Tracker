
class segment:
    def __init__(self,image,row,col):
        self.row = row
        self.col = col
        self.image = image
        self.area = image.shape[0] * image.shape[1]
