from pydantic import BaseModel



class ReviewText(BaseModel):
    resturant_name: str
    alias: str
    text: str
    date: str
    is_closed: bool
    address: str
    rating: float


    def clean(cls):
        pass

    def tokenize(cls):
        pass




