from pydantic import BaseModel

# Repersent data structure of csv file
class SCDataset(BaseModel):
    title: str
    url: str
    disease_type: str
    content: str
    causes: str