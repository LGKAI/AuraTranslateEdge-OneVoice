from pathlib import Path
import json,sys,hashlib,pycrfsuite,random
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'s2s_edge'))
from s2s_edge.identifier_tagger import tokens,features,IdentifierTagger
source=r/'research/tts_format_benchmark/train_cases.json';data=json.loads(source.read_text())
# Extra training examples diversify character shapes and groupings. These are
# supervised training data generation, not inference format handlers.
rng=random.Random(73129);aug=[]
for c in data:
 if c['label'] not in ('code','telephone','measure','date','time','money'):continue
 for _ in range(5):
  if c['label']=='telephone':
   value='0'+''.join(str(rng.randrange(10)) for _ in range(rng.randint(6,12)))
   if rng.random()<.5:value=' '.join(value[i:i+3] for i in range(0,len(value),3))
  elif c['label']=='code':
   value=''.join(str(rng.randrange(10)) for _ in range(rng.randint(1,9)))
   if rng.random()<.6:value=''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(rng.randint(2,3)))+value
  elif c['label']=='date':value=f'{rng.randint(1,28)}/{rng.randint(1,12)}/{rng.randint(1950,2099)}'
  elif c['label']=='time':value=f'{rng.randint(0,23)}{rng.choice([":","h"])}{rng.randint(0,59):02d}'
  elif c['label']=='money':value=f'{rng.randint(10,99)}.000 đồng' if c['lang']=='vi' else f'${rng.randint(1,99)}.{rng.randint(0,99):02d}'
  else:value=str(rng.randint(10,90))+rng.choice([',','.'])+str(rng.randrange(10))+'°C'
  aug.append({**c,'text':c['text'].replace(c['target'],value),'target':value})
# Vary natural-language context using new generated values, never evaluation labels.
for lang,templates in {'vi':['Xin gọi số {} để liên hệ.','Chuyến bay {} khởi hành.','Mã ZIP của tôi là {}.','Mã địa chỉ của tôi là {}.'], 'en':['Call {} to contact us.','My flight is {}.','My ZIP code is {}.']}.items():
 for t in templates:
  for _ in range(120):
   value='0'+''.join(str(rng.randrange(10)) for _ in range(9))
   if 'bay' in t or 'flight' in t:value=''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(2))+''.join(str(rng.randrange(10)) for _ in range(rng.randint(1,6)))
   aug.append({'lang':lang,'text':t.format(value),'target':value,'label':'telephone' if 'gọi' in t or 'Call' in t else 'code'})
for lang in ('vi','en'):
 for _ in range(1400):
  phone='0'+''.join(str(rng.randrange(10)) for _ in range(9))
  if rng.random()<.5:phone=' '.join([phone[:4],phone[4:7],phone[7:]])
  code=''.join(rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(2))+''.join(str(rng.randrange(10)) for _ in range(rng.randint(2,5)))
  zipvalue=''.join(str(rng.randrange(10)) for _ in range(5))
  timevalue=f'{rng.randint(0,23)}:{rng.randint(0,59):02d}'
  date=f'{rng.randint(1,28)}/{rng.randint(1,12)}/{rng.randint(1950,2099)}'
  if lang=='vi':text=f'Xin gọi số {phone}. Chuyến bay {code}, mã {rng.choice(["ZIP","bưu chính","địa chỉ"])} là {zipvalue}. Họp lúc {timevalue} ngày {date}. Giá vé {rng.randint(10,99)}.000 đồng, nhiệt độ {rng.randint(10,39)},{rng.randrange(10)}°C.'
  else:text=f'Call {phone}. Flight {code}, my ZIP code is {zipvalue}. Meet at {timevalue} on {date}. The ticket costs ${rng.randint(10,99)}.{rng.randint(0,99):02d}, temperature {rng.randint(10,39)}.{rng.randrange(10)} degrees Celsius.'
  aug.append({'lang':lang,'text':text,'entities':[phone,code,zipvalue]})
# Learn spoken identifier spans too, so MT cannot translate a phone digit
# such as Vietnamese "năm" into the English word "years".
spoken={'vi':'không một hai ba bốn năm sáu bảy tám chín'.split(),'en':'zero one two three four five six seven eight nine'.split()}
for lang in ('vi','en'):
 for _ in range(900):
  value=' '.join(spoken[lang][rng.randrange(10)] for _ in range(rng.randint(5,12)))
  if lang=='vi':
   text=rng.choice(['Xin gọi số {} để liên hệ.','Gọi số {} nhé.','Điện thoại của tôi là {}.','Mã bưu chính là {}.']).format(value)
  else:text=rng.choice(['Call {} to contact us.','My phone number is {}.','My ZIP code is {}.']).format(value)
  aug.append({'lang':lang,'text':text,'entities':[value]})
  # Same number words in a quantity context are not identifiers.
  negative=rng.choice(['Có {} người.','Giá vé {} đồng.','Nhiệt độ là {} độ C.']) if lang=='vi' else rng.choice(['There are {} people.','The ticket costs {} dollars.','The temperature is {} degrees Celsius.'])
  aug.append({'lang':lang,'text':negative.format(value),'entities':[]})
data+=aug
trainer=pycrfsuite.Trainer(verbose=False)
for c in data:
 text=c['text'];targets=c.get('entities', [c['target']] if 'target' in c and c['label'] in ('code','telephone') else []);bounds=[(text.index(v),text.index(v)+len(v)) for v in targets];ms=tokens(text);labels=[];inside=False
 for m in ms:
  yes=any(m.start()<b and m.end()>a for a,b in bounds)
  labels.append(('I' if inside else 'B') if yes else 'O');inside=yes
 trainer.append([features(ms,i,c['lang']) for i in range(len(ms))],labels)
trainer.set_params({'c1':.05,'c2':.05,'max_iterations':120,'feature.possible_transitions':True})
out=r/'models/normalization';out.mkdir(parents=True,exist_ok=True);trainer.train(str(out/'identifier.crfsuite'))
(out/'identifier_training.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'sentences':len(data),'augmentation_seed':73129,'augmentation':'identifier lengths/grouping/alphabetic prefixes plus measurement negatives','test_cases_used_for_training':False,'labels':'BIO for identifiers vs other spans','tokenization':'whitespace only; no time/phone/money patterns'},indent=2))
m=IdentifierTagger(out/'identifier.crfsuite')
for lang,s in [('vi','Nhiệt độ là 36,8°C.'),('vi','Xin gọi số 0912 345 678. Chuyến bay VN123, mã ZIP 67297.'),('en','My phone number is 0908246135.'),('en','The temperature is 36.8 degrees Celsius.')]:print(s,m(s,lang))
