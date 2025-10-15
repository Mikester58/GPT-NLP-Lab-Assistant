"""
File to maintain the model and ensure functionality.
"""
import ollama
import tqdm

#Helper Functions
def CheckModelAvailability(modelName: str) -> bool:
    def CheckLocalAvailability(modelName: str) -> bool:
        try:
            ollama.show()
            return True
        except ollama.ResponseError:
            return False
    

def GetListOfModels() -> str:

def PullModel(modelName: str) -> None:
    if CheckModelAvailability(modelName=modelName):
        digesting, bars = "", {}
        for prog in ollama.pull(modelName, stream=True):
            digesting = prog.get("digest", "")
