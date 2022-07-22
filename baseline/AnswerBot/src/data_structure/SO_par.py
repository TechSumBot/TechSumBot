class SO_Par:
    __slots__ = 'ans_id', 'body', 'score', 'que_id'

    def __init__(self, ans_id, body, score, que_id):
        self.id = ans_id
        self.body = body
        self.score = score
        self.que_id = que_id
