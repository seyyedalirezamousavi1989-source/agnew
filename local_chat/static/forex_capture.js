/* Forex chat can request a screen capture in response to an explicit user message. Browser permission remains mandatory. */
(() => {
  const overlay = document.getElementById('forexCaptureOverlay');
  if (!overlay) return;
  const video = document.getElementById('forexCaptureVideo');
  const canvas = document.getElementById('forexCaptureCanvas');
  const ctx = canvas.getContext('2d');
  const placeholder = document.getElementById('capturePlaceholder');
  const hint = document.getElementById('captureHint');
  const saveButton = document.getElementById('saveCapture');
  let stream = null, frozen = false, tool = 'crop', start = null, current = null, crop = null, marks = [], undoStack = [], counter = 0;
  let frame = null, captureResolve = null;
  const setTool = (next) => { tool = next; overlay.querySelectorAll('[data-tool]').forEach(b => b.classList.toggle('active', b.dataset.tool === tool)); };
  function close(result = false) { if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; video.srcObject = null; overlay.hidden = true; frozen = false; frame = null; crop = null; marks = []; undoStack = []; canvas.hidden = true; video.hidden = false; placeholder.hidden = false; saveButton.disabled = true; const done = captureResolve; captureResolve = null; if (done) done(result); }
  document.getElementById('closeForexCapture').addEventListener('click', () => close(false));
  overlay.addEventListener('click', e => { if (e.target === overlay) close(false); });
  async function waitForDesktopBridge(timeoutMs = 1800) {
    if (window.pywebview?.api?.capture_screen) return window.pywebview.api;
    // pywebview injects its API asynchronously. Wait briefly before deciding
    // that this is a plain browser session.
    await new Promise(resolve => {
      let settled = false;
      const finish = () => { if (!settled) { settled = true; window.removeEventListener('pywebviewready', finish); resolve(); } };
      window.addEventListener('pywebviewready', finish, { once: true });
      setTimeout(finish, timeoutMs);
    });
    return window.pywebview?.api?.capture_screen ? window.pywebview.api : null;
  }

  async function showNativeCapture(result) {
    if (!result?.ok) { if (result?.cancelled) return false; throw new Error(result?.error || 'تصویربرداری بومی ناموفق بود.'); }
    const binary = atob(result.data); const bytes = new Uint8Array(binary.length);
    for (let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
    const blob = new Blob([bytes], {type: result.mime || 'image/png'});
    const bitmap = await createImageBitmap(blob);
    frame = document.createElement('canvas'); frame.width=bitmap.width; frame.height=bitmap.height;
    frame.getContext('2d').drawImage(bitmap,0,0); bitmap.close?.();
    canvas.width=frame.width; canvas.height=frame.height; canvas.hidden=false; video.hidden=true;
    placeholder.hidden=true; overlay.hidden=false; frozen=true; crop=null; marks=[]; undoStack=[]; counter=0; saveButton.disabled=false;
    hint.textContent='تصویر دسکتاپ ثبت شد. محدودهٔ نمودار XAUUSD را برش بده و در صورت نیاز نقاط مهم را علامت‌گذاری کن.';
    draw(); return new Promise(resolve => { captureResolve=resolve; });
  }

  window.startForexCapture = async function startForexCapture() {
    if (typeof forexEnabled !== 'undefined' && !forexEnabled) throw new Error('ابتدا حالت تحلیل فارکس را روشن کن.');
    // Prefer the native Windows bridge in the packaged desktop app. WebView2
    // may expose getDisplayMedia while still rejecting its picker, so testing
    // only for the browser API incorrectly bypasses the working native path.
    const desktopApi = await waitForDesktopBridge();
    if (desktopApi) return showNativeCapture(await desktopApi.capture_screen());
    if (!navigator.mediaDevices?.getDisplayMedia) {
      throw new Error('برای تصویربرداری، LocalChat را از فایل دسکتاپ LocalChat.exe اجرا کن یا از مرورگری استفاده کن که انتخاب پنجره/صفحه را پشتیبانی می‌کند.');
    }
    return new Promise((resolve, reject) => {
      captureResolve = resolve;
      // Called directly from the user's Send action so the browser can show its native permission picker.
      navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 8 }, audio: false }).then(async selected => {
        stream = selected; video.srcObject = stream; await video.play(); overlay.hidden = false; frozen = false;
        video.hidden = false; canvas.hidden = true; placeholder.hidden = true;
        hint.textContent = 'فریم را ثبت کن؛ سپس ناحیهٔ موردنظر را برش بده. برای حفظ حریم خصوصی، انتخاب پنجره/مانیتور با اجازهٔ خودت انجام می‌شود.';
        stream.getVideoTracks()[0]?.addEventListener('ended', () => close(false), { once: true });
      }).catch(err => { captureResolve = null; if (err.name === 'AbortError') resolve(false); else reject(new Error(`دسترسی به تصویر صفحه ممکن نشد: ${err.message}`)); });
    });
  };
  function draw() {
    if (!frame) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
    if (crop) {
      ctx.save(); ctx.strokeStyle = '#12a879'; ctx.lineWidth = Math.max(2, canvas.width / 700); ctx.setLineDash([8, 5]);
      ctx.strokeRect(crop.x, crop.y, crop.w, crop.h); ctx.restore();
    }
    marks.forEach(m => {
      ctx.save(); ctx.strokeStyle = '#ffb020'; ctx.fillStyle = '#ffb020'; ctx.lineWidth = Math.max(3, canvas.width / 450); ctx.lineCap = 'round';
      if (m.type === 'pen' || m.type === 'line' || m.type === 'arrow') {
        ctx.beginPath(); ctx.moveTo(m.x1, m.y1); ctx.lineTo(m.x2, m.y2); ctx.stroke();
        if (m.type === 'arrow') { const a=Math.atan2(m.y2-m.y1,m.x2-m.x1), h=Math.max(14,canvas.width/45); ctx.beginPath(); ctx.moveTo(m.x2,m.y2); ctx.lineTo(m.x2-h*Math.cos(a-.45),m.y2-h*Math.sin(a-.45)); ctx.lineTo(m.x2-h*Math.cos(a+.45),m.y2-h*Math.sin(a+.45)); ctx.closePath(); ctx.fill(); }
      } else if (m.type === 'number') { const r=Math.max(15,canvas.width/55); ctx.beginPath(); ctx.arc(m.x,m.y,r,0,Math.PI*2); ctx.fill(); ctx.fillStyle='#202020'; ctx.font=`bold ${r*1.2}px sans-serif`; ctx.textAlign='center'; ctx.textBaseline='middle'; ctx.fillText(String(m.n),m.x,m.y); }
      ctx.restore();
    });
    if (current && start && tool !== 'crop') { ctx.save(); ctx.strokeStyle='#ffb020'; ctx.lineWidth=Math.max(3,canvas.width/450); ctx.beginPath();ctx.moveTo(start.x,start.y);ctx.lineTo(current.x,current.y);ctx.stroke();ctx.restore(); }
  }
  document.getElementById('captureFrame').addEventListener('click', () => {
    if (!video.videoWidth) return;
    frame = document.createElement('canvas'); frame.width = video.videoWidth; frame.height = video.videoHeight;
    frame.getContext('2d').drawImage(video,0,0); canvas.width=frame.width;canvas.height=frame.height;
    video.hidden=true;canvas.hidden=false;placeholder.hidden=true;frozen=true;crop=null;marks=[];undoStack=[];counter=0;saveButton.disabled=false;draw();
    hint.textContent='برای انتخاب محدوده، ابزار «برش ناحیه» را انتخاب و روی تصویر بکش. سپس با قلم، خط، فلش یا شماره نقاط مهم را مشخص کن.';
  });
  overlay.querySelectorAll('[data-tool]').forEach(b=>b.addEventListener('click',()=>setTool(b.dataset.tool)));
  document.getElementById('resetCapture').addEventListener('click',()=>{crop=null;marks=[];undoStack=[];draw();});
  document.getElementById('undoCapture').addEventListener('click',()=>{if(undoStack.length){const prev=undoStack.pop();crop=prev.crop;marks=prev.marks;draw();}});
  function pos(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*canvas.width/r.width,y:(e.clientY-r.top)*canvas.height/r.height};}
  canvas.addEventListener('pointerdown',e=>{if(!frozen)return;canvas.setPointerCapture(e.pointerId);start=pos(e);current=start;undoStack.push({crop:crop&&{...crop},marks:marks.map(m=>({...m}))});});
  canvas.addEventListener('pointermove',e=>{if(!start)return;current=pos(e);if(tool==='crop'){const x=Math.min(start.x,current.x),y=Math.min(start.y,current.y);crop={x,y,w:Math.abs(current.x-start.x),h:Math.abs(current.y-start.y)};}draw();});
  canvas.addEventListener('pointerup',e=>{if(!start)return;const end=pos(e);if(tool==='pen'||tool==='line'||tool==='arrow')marks.push({type:tool,x1:start.x,y1:start.y,x2:end.x,y2:end.y});else if(tool==='number'){counter++;marks.push({type:'number',x:end.x,y:end.y,n:counter});}start=null;current=null;draw();});
  saveButton.addEventListener('click',async()=>{
    if(!frame||typeof addOneFile!=='function')return;
    saveButton.disabled=true;
    try{
      const region = crop&&crop.w>8&&crop.h>8 ? crop : {x:0,y:0,w:frame.width,h:frame.height};
      const out=document.createElement('canvas');out.width=Math.max(1,Math.round(region.w));out.height=Math.max(1,Math.round(region.h));
      const outCtx=out.getContext('2d');
      outCtx.drawImage(frame,region.x,region.y,region.w,region.h,0,0,out.width,out.height);
      const sx=out.width/region.w, sy=out.height/region.h;
      marks.forEach(m=>{outCtx.save();outCtx.strokeStyle='#ffb020';outCtx.fillStyle='#ffb020';outCtx.lineWidth=Math.max(3,out.width/450);outCtx.lineCap='round';
        if(m.type==='pen'||m.type==='line'||m.type==='arrow'){const x1=(m.x1-region.x)*sx,y1=(m.y1-region.y)*sy,x2=(m.x2-region.x)*sx,y2=(m.y2-region.y)*sy;outCtx.beginPath();outCtx.moveTo(x1,y1);outCtx.lineTo(x2,y2);outCtx.stroke();if(m.type==='arrow'){const a=Math.atan2(y2-y1,x2-x1),h=Math.max(14,out.width/45);outCtx.beginPath();outCtx.moveTo(x2,y2);outCtx.lineTo(x2-h*Math.cos(a-.45),y2-h*Math.sin(a-.45));outCtx.lineTo(x2-h*Math.cos(a+.45),y2-h*Math.sin(a+.45));outCtx.closePath();outCtx.fill();}}
        else if(m.type==='number'){const x=(m.x-region.x)*sx,y=(m.y-region.y)*sy,r=Math.max(15,out.width/55);outCtx.beginPath();outCtx.arc(x,y,r,0,Math.PI*2);outCtx.fill();outCtx.fillStyle='#202020';outCtx.font=`bold ${r*1.2}px sans-serif`;outCtx.textAlign='center';outCtx.textBaseline='middle';outCtx.fillText(String(m.n),x,y);}outCtx.restore();});
      const blob=await new Promise(resolve=>out.toBlob(resolve,'image/jpeg',0.9));
      if(!blob)throw new Error('ساخت تصویر ناموفق بود.');
      await addOneFile(new File([blob],`forex-chart-${Date.now()}.jpg`,{type:'image/jpeg'}));
      close(true); showToast('تصویر نمودار به پیوست همین پیام اضافه شد و برای تحلیل ارسال می‌شود.');
    }catch(err){showToast(err.message||'افزودن تصویر ناموفق بود.');saveButton.disabled=false;}
  });
})();
