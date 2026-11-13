"""Learned identifier detection. Whitespace tokenization, no format regex rules."""
import re,unicodedata
from pathlib import Path

def tokens(text):return list(re.finditer(r'\S+',text))
def lexical(s):return ''.join(c.casefold() for c in s if c.isalnum())
def features(ms,i,lang):
 s=ms[i].group();shape=''.join('D' if c.isdecimal() else 'U' if c.isupper() else 'l' if c.isalpha() else c for c in s)
 out={'bias':1.,'lang':lang,'shape':shape,'length':min(len(s),20),'has_digit':any(c.isdecimal() for c in s),'has_alpha':any(c.isalpha() for c in s),'upper':s.isupper()}
 for d in range(-4,5):
  if 0<=i+d<len(ms):
   w=ms[i+d].group();out[f'context{d}']=''.join('D' if c.isdecimal() else c.casefold() for c in w if c.isalnum())
 return out

class IdentifierTagger:
 def __init__(self,path):
  import pycrfsuite
  self.t=pycrfsuite.Tagger();self.t.open(str(path))
 def __call__(self,text,lang):
  ms=tokens(text)
  if not ms:return []
  ys=self.t.tag([features(ms,i,lang) for i in range(len(ms))]);out=[];i=0
  while i<len(ms):
   if ys[i]=='O':i+=1;continue
   a=ms[i].start();j=i+1
   while j<len(ms) and ys[j]=='I':j+=1
   b=ms[j-1].end()
   while a<b and not text[a].isalnum():a+=1
   while b>a and not text[b-1].isalnum():b-=1
   if b>a:out.append(text[a:b])
   i=j
  return out
