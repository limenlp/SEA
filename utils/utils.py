import re




def ans_extractor(ans: str):
    """
    Extract the answer from the answer string.
    """
    matches1 = re.findall(r'\\box\{(.*?)\}', ans)
    matches2 = re.findall(r'\\boxed\{(.*?)\}', ans)
    if len(matches1) > 0:
        return matches1[-1].lower()
    if len(matches2) > 0:
        return matches2[-1].lower()
    return ans
    
    
def judge_ans(ans, gt):
    gt = gt.lower()
    if ans == "":
        return False
    
    if ans == gt:
        return True
    
    if ans[0] in ['a', 'b', 'c', 'd'] and gt[0] in ['a', 'b', 'c', 'd'] and ans[0] == gt[0]:
        return True
    
    if ans.startswith("a: ") or ans.startswith("b: ") or ans.startswith("c: ") or ans.startswith("d: "):
        ans = ans[3:]
    if gt.startswith("a: ") or gt.startswith("b: ") or gt.startswith("c: ") or gt.startswith("d: "):
        gt = gt[3:]
        
    if ans == gt:
        return True
    
    return False