// Capture actual microphone PCM. No speech recognition or synthetic input.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(){super();this.buffer=new Float32Array(2048);this.count=0;this.port.onmessage=e=>{if(e.data==='flush'){this.flush();this.port.postMessage({flushed:true})}}}
  flush(){if(this.count){const data=this.buffer.slice(0,this.count);this.port.postMessage({samples:data},[data.buffer]);this.count=0}}
  process(inputs){const x=inputs[0]?.[0];if(x){for(let i=0;i<x.length;i++){this.buffer[this.count++]=x[i];if(this.count===this.buffer.length)this.flush()}}return true}
}
registerProcessor('capture',CaptureProcessor);
