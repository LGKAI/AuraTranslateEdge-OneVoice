'use strict';
const $=id=>document.getElementById(id), names={en:'English',vi:'Tiếng Việt'};
let src='en',recording=false,busy=false,context,stream,node,sourceNode,mute,chunks=[],started,timer,flushed,blob,previewURL,finishing=false;
function status(text,kind=''){ $('status').textContent=text;$('status').className='status '+kind }
function controls(){for(const id of ['en','vi','choose','retry'])$(id).disabled=recording||busy||finishing;$('record').disabled=busy||finishing;for(const id of ['en','vi']){$(id).classList.toggle('selected',src===id);$(id).setAttribute('aria-pressed',String(src===id))}}
function direction(lang){src=lang;blob=null;$('input-preview').hidden=true;$('record-hint').textContent=`Nói bằng ${lang==='en'?'tiếng Anh':'tiếng Việt'} · Tối đa 30 giây`;controls()}
$('en').onclick=()=>direction('en');$('vi').onclick=()=>direction('vi');
function wav(samples,rate){const buffer=new ArrayBuffer(44+samples.length*2),v=new DataView(buffer);const str=(a,t)=>{for(let i=0;i<t.length;i++)v.setUint8(a+i,t.charCodeAt(i))};str(0,'RIFF');v.setUint32(4,36+samples.length*2,true);str(8,'WAVE');str(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,rate,true);v.setUint32(28,rate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);str(36,'data');v.setUint32(40,samples.length*2,true);for(let i=0;i<samples.length;i++){const s=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,s<0?s*32768:s*32767,true)}return new Blob([buffer],{type:'audio/wav'})}
async function encode(buffer){if(buffer.duration>30.1)throw new Error('File dài hơn 30 giây. Hãy chọn một đoạn ngắn hơn.');if(buffer.duration<.25)throw new Error('Bản thu quá ngắn. Hãy nói ít nhất một câu.');const off=new OfflineAudioContext(1,Math.ceil(buffer.duration*16000),16000),s=off.createBufferSource();s.buffer=buffer;s.connect(off.destination);s.start();const res=await off.startRendering();return wav(res.getChannelData(0),16000)}
function preview(b){blob=b;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(b);$('input-audio').src=previewURL;$('input-preview').hidden=false}
function recordingUI(on){$('orbit').classList.toggle('recording',on);$('mic-icon').toggleAttribute('hidden',on);$('stop-icon').hidden=!on;$('clock').hidden=!on;$('record').setAttribute('aria-label',on?'Dừng thu và dịch':'Bắt đầu thu âm');$('record-label').textContent=on?'Đang nghe… Bấm để dừng và dịch':'Bấm mic để bắt đầu';if(!on)$('meter').firstElementChild.style.width='0%';controls()}
async function start(){
 if(!navigator.mediaDevices?.getUserMedia||!window.AudioWorkletNode)throw new Error('Hãy mở trang bằng Chrome hoặc Safari trên máy này để dùng mic.');
 document.querySelectorAll('audio').forEach(a=>a.pause());blob=null;chunks=[];$('input-preview').hidden=true;finishing=true;controls();status('Đang xin quyền dùng mic…');
 try{
  stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true},video:false});context=new AudioContext();await context.resume();await context.audioWorklet.addModule('/recorder.js');sourceNode=context.createMediaStreamSource(stream);node=new AudioWorkletNode(context,'capture');mute=context.createGain();mute.gain.value=0;sourceNode.connect(node);node.connect(mute).connect(context.destination);
  node.port.onmessage=e=>{if(e.data.flushed){flushed?.();return}const x=e.data.samples;if(x){chunks.push(x);let p=0;for(const v of x)p=Math.max(p,Math.abs(v));$('meter').firstElementChild.style.width=Math.min(100,p*220)+'%'}};
  started=performance.now();recording=true;finishing=false;recordingUI(true);status('Bạn đang thu âm. Bấm mic lần nữa để dừng và dịch.','working');
  const tick=()=>{const s=Math.floor((performance.now()-started)/1000);$('clock').textContent=`00:${String(Math.min(s,30)).padStart(2,'0')} / 00:30`;if(s>=30)stop().catch(showError)};tick();timer=setInterval(tick,200);
 }catch(e){stream?.getTracks().forEach(t=>t.stop());await context?.close().catch(()=>{});finishing=false;controls();throw e}
}
async function stop(){
 if(!recording||finishing)return;recording=false;finishing=true;clearInterval(timer);controls();sourceNode.disconnect();
 await Promise.race([new Promise(resolve=>{flushed=resolve;node.port.postMessage('flush')}),new Promise(resolve=>setTimeout(resolve,1000))]);
 const rate=context.sampleRate;stream.getTracks().forEach(t=>t.stop());node.disconnect();mute.disconnect();await context.close();
 const total=chunks.reduce((a,x)=>a+x.length,0),n=Math.min(total,Math.floor(rate*30)),all=new Float32Array(n);let offset=0;for(const x of chunks){const part=x.subarray(0,Math.max(0,n-offset));all.set(part,offset);offset+=part.length;if(offset>=n)break}chunks=[];
 if(n<rate*.25)throw new Error('Bản thu quá ngắn. Hãy nói ít nhất một câu.');
 const input=new AudioBuffer({length:all.length,numberOfChannels:1,sampleRate:rate});input.copyToChannel(all,0);
 try{preview(await encode(input));finishing=false;recordingUI(false);await submit()}finally{finishing=false;recordingUI(false)}
}
function showError(e){finishing=false;busy=false;recording=false;clearInterval(timer);recordingUI(false);const text=e.name==='NotAllowedError'?'Mic chưa được cho phép. Bật quyền mic cho localhost trong trình duyệt rồi thử lại.':e.name==='NotFoundError'?'Không tìm thấy mic. Hãy kết nối hoặc chọn một mic trên máy.':e.message||String(e);status(text,'error')}
$('record').onclick=()=>{(recording?stop():start()).catch(showError)};
$('choose').onclick=()=>{$('file').value='';$('file').click()};
$('file').onchange=async()=>{const file=$('file').files[0];if(!file)return;finishing=true;controls();status('Đang đọc file audio…','working');let ctx;try{ctx=new AudioContext();const data=await ctx.decodeAudioData(await file.arrayBuffer());preview(await encode(data));finishing=false;controls();await submit()}catch(e){showError(e)}finally{await ctx?.close().catch(()=>{});finishing=false;controls()}};
$('retry').onclick=()=>submit().catch(showError);
async function submit(){
 if(!blob||busy)return;busy=true;controls();const lang=src;status('Đang gửi bản thu…','working');
 try{
  const response=await fetch('/api/jobs?src='+encodeURIComponent(lang),{method:'POST',headers:{'Content-Type':'audio/wav'},body:blob});const body=await response.json();if(!response.ok)throw new Error(body.error||'Không gửi được audio.');
  let result;
  for(;;){await new Promise(r=>setTimeout(r,800));const res=await fetch('/api/jobs/'+body.id);if(!res.ok)throw new Error('Mất kết nối với bản thu. Hãy thử lại.');result=await res.json();if(result.status==='done')break;if(result.status==='error')throw new Error(result.error||'Pipeline chưa xử lý được bản thu.');status(result.stage+(result.segments_total>1?` (${result.segment||1}/${result.segments_total} đoạn)`:'')+' Thời gian đầu có thể lâu hơn.','working')}
  const card=render(result);status(`Hoàn tất trong ${result.elapsed_seconds.toFixed(1)} giây. Bấm phát để nghe hoặc thu câu tiếp theo.`);try{await card.querySelector('.output').play()}catch{}
 }finally{busy=false;controls()}
}
function render(r){
 const card=document.createElement('article');card.className='result panel';
 card.innerHTML='<div class="result-head"><h2></h2><span class="duration"></span></div><div class="transcript-grid"><div><span class="overline">MIC ĐÃ NHẬN ĐƯỢC</span><p class="asr"></p></div><div><span class="overline">BẢN DỊCH</span><p class="translation translated"></p></div></div><div class="output-label"><span class="overline">NGHE BẢN DỊCH</span><a class="download" download="translation.wav">Tải audio ↗</a></div><audio class="output" controls></audio><details><summary>Xem cách đọc và bản thu gốc</summary><p class="tts"></p><span class="overline">AUDIO ĐẦU VÀO</span><audio class="original" controls></audio><a class="detail" target="_blank" rel="noopener">Xem kết quả từng bước ↗</a></details>';
 card.querySelector('h2').textContent=names[r.src]+' → '+names[r.tgt];card.querySelector('.duration').textContent=`${r.input_seconds.toFixed(1)}s đầu vào · ${r.elapsed_seconds.toFixed(1)}s xử lý`;card.querySelector('.asr').textContent=r.asr;card.querySelector('.translation').textContent=r.translation;card.querySelector('.tts').textContent='TTS đọc: '+r.tts_text;card.querySelector('.output').src=r.output_url;card.querySelector('.original').src=r.input_url;card.querySelector('.download').href=r.output_url;card.querySelector('.detail').href=r.detail_url;$('results').prepend(card);while($('results').children.length>5)$('results').lastElementChild.remove();return card
}
async function health(){try{const r=await fetch('/api/health');if(!r.ok)throw 0;const h=await r.json();$('connection').textContent=h.pending?'● Máy đang xử lý':'● Máy đang sẵn sàng'}catch{$('connection').textContent='○ Chưa kết nối máy xử lý'}}health();setInterval(health,10000);
window.addEventListener('beforeunload',()=>stream?.getTracks().forEach(t=>t.stop()));controls();
