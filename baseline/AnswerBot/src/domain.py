class Question:

    def __init__(self,id,title,body,tags):
        self.id = id
        self.title = title
        self.body = body
        self.tags = tags
        self.title_words = None
        self.matrix = None
        self.idf_vector = None