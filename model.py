"""
File to maintain the model and ensure functionality of ollama integration.
Provides:
- CheckModelAvailability(modelName) -> bool
- GetListOfModels() -> list[str]
- PullModel(modelName) -> bool
"""
import ollama
from tqdm import tqdm
from typing import List

#Helper Functions
def CheckLocalAvailability(modelName: str) -> bool:
    """
    Return True if the model is available locally via ollama
    """
    try:
        ollama.show(model=modelName)
        return True
    except (ollama.ResponseError, Exception):
        return False
        
def CheckModelAvailability(modelName: str) -> bool:
    """
    Ensure the specified model is available locally, attempt to pull it if not.
    Return True when the model becomes available locally (with or without pulling)
    Return False if the model couldnt be found nor pulled.
    """

    if CheckLocalAvailability(modelName):
        return True
    
    try:
        pulled=PullModel(modelName)
    except Exception:
        pulled = False
    
    if pulled & CheckLocalAvailability(modelName):
        return True
    
    return False
    

def GetListOfModels() -> List[str]:
    """
    Return a list of model names from Ollama service.
    If service cant be contacted, return an empty list
    """
    try:
        response = ollama.list()
        models = response.get("models", [])
        return [m.get("model") for m in models if isinstance(m, dict) and "model" in m]
    except Exception:
        return []


def PullModel(modelName: str) -> None:
    """
    Attempt to pull modelName from Ollama hub.
    Stream progress and returns True or False depending on success.
    """
    try:
        currDigest = ""
        bars = {}

        for progress in ollama.pull(modelName, stream=True):
            digest = progress.get("digest", "")
            
            #If bar changes close the digest
            if digest != currDigest and currDigest in bars:
                try:
                    bars[currDigest].close()
                except Exception:
                    pass
            
            #clean status text
            if not digest:
                status = progress.get("status")
                if status:
                    print(status)
                continue
            
            #make new tqdm bar if required
            if digest not in bars and (total := progress.get("total")):
                bars[digest] = tqdm(total=total, desc=f"pulling {digest[7:19]}", unit="B", unit_scale=True)
            
            if completed := progress.get("completed"):
                try:
                    bars[digest].update(completed - bars[digest].n)
                except Exception:
                    #if out of sync ignore it & continue
                    pass
            
            currDigest = digest

        for bar in bars.values():
            try:
                bar.close()
            except Exception:
                pass
        print(f"Successfully pulled model {modelName}!")
        return True
    except Exception as e:
        print(f"Failed to pull model {modelName}: {e}")
        return False
