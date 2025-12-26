let handLandmarks;
let myCapture;

// 예측 변수
let predictedNote = "";
let predictedConfidence = 0;
let lastPredictionRequestMs = 0;
let isPredicting = false;
const PREDICT_INTERVAL_MS = 1200;
const PREDICT_ERROR_PAUSE_MS = 4000;
const PREDICT_API_URL = "/api/predict";

// UI
let handGraphics;
let boxW = 800;
let boxH = 600;

// 손 트래킹 config
let trackingConfig = {
  doAcquireHandLandmarks: true,
  poseModelLiteOrFull: "full",
  cpuOrGpuString: "GPU",
  maxNumHands: 2,
};

// 🔊 Tone.js 관련 변수
let synth;
let audioStarted = false;


const NOTE_MAP = {
  "Do": "C4",
  "Re": "D4",
  "Mi": "E4",
  "Fa": "F4",
  "Sol": "G4",
  "La": "A4",
  "Ti": "B4",
};

// 주파수 매핑 (Hz)
const FREQ_MAP = {
  "Do": 261.63,
  "Re": 293.66,
  "Mi": 329.63,
  "Fa": 349.23,
  "Sol": 392.00,
  "La": 440.00,
  "Ti": 493.88,
};

// 파티클 시스템
let particles = [];

async function preload() {
  if (typeof preloadTracker === "function") {
    preloadTracker();
  }
}

function setup() {
  createCanvas(windowWidth, windowHeight);
  pixelDensity(1);

  myCapture = createCapture(VIDEO);
  myCapture.size(320, 240);
  myCapture.hide();
  handGraphics = createGraphics(boxW, boxH);

  if (typeof initiateTracking === "function") {
    initiateTracking();
  }

  // 🔊 Tone Synth 준비 (아직 AudioContext는 잠겨 있음)
  synth = new Tone.Synth().toDestination();

  // 🔊 반드시 사용자 클릭 시만 AudioContext 활성화 가능
  window.addEventListener("click", async () => {
    if (!audioStarted) {
      await Tone.start();
      console.log("🔊 AudioContext started!");
      audioStarted = true;
    }
  });
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

// ----------------------------
// 프레임 캡처 → 예측 요청
// ----------------------------
async function captureFrameAsBlob() {
  return new Promise((resolve, reject) => {
    const video = myCapture?.elt;
    if (!video || video.readyState < 2) 
      return reject("video-not-ready");

    const w = video.videoWidth || 224;
    const h = video.videoHeight || 224;

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);

    canvas.toBlob((blob) => {
      if (!blob) reject("blob-failed");
      else resolve(blob);
    }, "image/jpeg", 0.9);
  });
}

async function requestPredictionIfNeeded() {
  const now = millis();
  if (isPredicting || now - lastPredictionRequestMs < PREDICT_INTERVAL_MS) return;

  lastPredictionRequestMs = now;
  isPredicting = true;

  try {
    const frameBlob = await captureFrameAsBlob();
    const form = new FormData();
    form.append("file", frameBlob, "frame.jpg");

    const res = await fetch(PREDICT_API_URL, { method: "POST", body: form });
    if (!res.ok) throw new Error(res.status);

    const data = await res.json();
    const pred = data?.prediction?.[0];

    if (pred?.label) {
      const newNote = pred.label;

      // 🔊 예측된 노트가 바뀌었을 때만 사운드 재생
      if (audioStarted) {
        playNoteSound(newNote);
      }

      predictedNote = newNote;
      predictedConfidence = pred.confidence ?? 0;
    }
  } catch (err) {
    console.error("prediction failed", err);
    lastPredictionRequestMs = millis() + PREDICT_ERROR_PAUSE_MS;
  } finally {
    isPredicting = false;
  }
}

// 🔊 예측된 note를 Tone.js 음으로 재생
function playNoteSound(label) {
  const toneNote = NOTE_MAP[label];
  if (!toneNote) return;

  synth.triggerAttackRelease(toneNote, "12n"); // 8n = 짧은 음
  
  // 파티클 생성 (손 중앙에서)
  createParticlesForNote(label);
}

// 파티클 생성 함수
function createParticlesForNote(label) {
  if (!handLandmarks?.landmarks?.length) return;
  
  const joints = handLandmarks.landmarks[0];
  
  const fingerTips = [4, 8, 12, 16, 20];
  
  const freq = FREQ_MAP[label] || 440;
  const numParticles = int(map(freq, 260, 500, 20, 40)); 
  
  for (let tipIndex of fingerTips) {
    const tip = joints[tipIndex];
    if (!tip) continue;
    
    let x = map(tip.x, 0, 1, width/2 + boxW/2, width/2 - boxW/2);
    let y = map(tip.y, 0, 1, height/2 - boxH/2, height/2 + boxH/2);
    
    for (let i = 0; i < numParticles / 5; i++) {
      particles.push(new FrequencyParticle(x, y, freq, label));
    }
  }
}

// 파티클 클래스
class FrequencyParticle {
  constructor(x, y, freq, label) {
    this.x = x;
    this.y = y;
    this.freq = freq;
    this.label = label;
    
    // 주파수에 따라 속도 변화
    const speed = map(freq, 260, 500, 1, 3);
    const angle = random(TWO_PI);
    this.vx = cos(angle) * speed;
    this.vy = sin(angle) * speed;
    
    // 주파수에 따라 색상 변화 (사이버틱한 네온 색상)
    this.hue = map(freq, 260, 500, 180, 320); // 청록 ~ 보라 범위
    this.alpha = 255;
    this.size = random(4, 20);
    this.lifespan = 600;
    this.rotation = random(TWO_PI);
    this.glowSize = this.size * 2;
  }
  
  update() {
    this.x += this.vx;
    this.y += this.vy;
    this.lifespan -= 2;
    this.alpha = this.lifespan * 0.4;
    
    // 회전
    this.rotation += 0.03;
    
    // 펄스 효과
    this.glowSize = this.size * 2 + sin(frameCount * 0.1) * 5;
  }
  
  display() {
    push();
    colorMode(HSB, 360, 100, 100, 255);
    
    translate(this.x, this.y);
    rotate(this.rotation);
    
    // 외부 글로우 (여러겹)
    for (let i = 3; i > 0; i--) {
      noFill();
      stroke(this.hue, 80, 100, this.alpha * 0.15 * i);
      strokeWeight(2);
      this.drawShape(this.glowSize * (1 + i * 0.3));
    }
    
    // 메인 아웃라인 (네온 효과)
    noFill();
    stroke(this.hue, 100, 100, this.alpha);
    strokeWeight(2);
    this.drawShape(this.size);
    
    // 내부 그리드/패턴
    stroke(this.hue, 60, 100, this.alpha * 0.6);
    strokeWeight(0.5);
    this.drawInnerPattern();
    
    pop();
  }
  
  drawShape(s) {
    switch(this.label) {
      case "Do":
        // 동심원
        circle(0, 0, s);
        circle(0, 0, s * 0.6);
        break;
      case "Re":
        // 기하학적 삼각형
        for (let i = 0; i < 3; i++) {
          let angle = TWO_PI / 3 * i;
          let x1 = cos(angle) * s/2;
          let y1 = sin(angle) * s/2;
          let x2 = cos(angle + TWO_PI/3) * s/2;
          let y2 = sin(angle + TWO_PI/3) * s/2;
          line(x1, y1, x2, y2);
        }
        break;
      case "Mi":
        rectMode(CENTER);
        square(0, 0, s);
        push();
        rotate(PI/4);
        square(0, 0, s * 0.7);
        pop();
        break;
      case "Fa":
        this.drawComplexStar(s);
        break;
      case "Sol":
        // 육각형 격자
        this.drawHexagonGrid(s);
        break;
      case "La":
        // 다이아몬드 체인
        for (let i = 0; i < 4; i++) {
          push();
          rotate(PI/2 * i);
          line(0, -s/2, 0, s/2);
          line(-s/4, 0, s/4, 0);
          pop();
        }
        circle(0, 0, s * 0.3);
        break;
      case "Ti":
        // 방사형 라인
        for (let i = 0; i < 12; i++) {
          let angle = TWO_PI / 12 * i;
          let x = cos(angle) * s/2;
          let y = sin(angle) * s/2;
          line(0, 0, x, y);
        }
        circle(0, 0, s);
        break;
      default:
        circle(0, 0, s);
    }
  }
  
  drawInnerPattern() {
    let s = this.size * 0.4;
    // 내부 그리드 패턴
    for (let i = -1; i <= 1; i++) {
      line(i * s/3, -s/2, i * s/3, s/2);
      line(-s/2, i * s/3, s/2, i * s/3);
    }
  }
  
  drawComplexStar(s) {
    let points = 8;
    for (let i = 0; i < points; i++) {
      let angle = TWO_PI / points * i;
      let x1 = cos(angle) * s/2;
      let y1 = sin(angle) * s/2;
      let x2 = cos(angle + PI/points) * s/4;
      let y2 = sin(angle + PI/points) * s/4;
      line(x1, y1, x2, y2);
      line(0, 0, x1, y1);
    }
    circle(0, 0, s * 0.2);
  }
  
  drawHexagonGrid(s) {
    let points = 6;
    for (let i = 0; i < points; i++) {
      let angle = TWO_PI / points * i;
      let x1 = cos(angle) * s/2;
      let y1 = sin(angle) * s/2;
      let x2 = cos(angle + TWO_PI/points) * s/2;
      let y2 = sin(angle + TWO_PI/points) * s/2;
      line(x1, y1, x2, y2);
      line(0, 0, x1, y1);
    }
  }
  
  isDead() {
    return this.lifespan <= 0;
  }
}

// ----------------------------
// 메인 draw()
// ----------------------------
function draw() {
  background(0);

  drawHandInBox();

  imageMode(CENTER);
  noFill();
//   stroke(100);
  rectMode(CENTER);
  rect(width / 2, height / 2, boxW, boxH);
  image(handGraphics, width / 2, height / 2);

  drawTopLabel();
  
  // 파티클 업데이트 및 그리기
  for (let i = particles.length - 1; i >= 0; i--) {
    particles[i].update();
    particles[i].display();
    if (particles[i].isDead()) {
      particles.splice(i, 1);
    }
  }

  requestPredictionIfNeeded();

  // 아직 오디오가 잠겨있다면 안내 텍스트 표시
  if (!audioStarted) {
    fill(255, 100, 150);
    textAlign(CENTER, CENTER);
    textSize(20);
    text("Click to Start", width / 2, height - 130);
  }
}


// ----------------------------
// 손 점만 그리기 (엄지/검지 끝 강조)
// ----------------------------
function drawHandInBox() {
  handGraphics.background(0);

  if (handLandmarks?.landmarks?.length > 0) {
    
    for (let h = 0; h < handLandmarks.landmarks.length; h++) {
      const joints = handLandmarks.landmarks[h];

		handGraphics.noStroke();

		for (let i = 0; i < joints.length; i++) {
		let p = joints[i];
		let x = map(p.x, 0, 1, handGraphics.width, 0);
		let y = map(p.y, 0, 1, 0, handGraphics.height);

		if (i === 4 || i === 8 || i === 12 || i === 16 || i === 20) {
			handGraphics.fill(200, 200, 120)
			handGraphics.circle(x, y, 24); 
		} else {
			handGraphics.fill(200)
			handGraphics.circle(x, y, 8);
		}
		}
   		drawThumbIndexHighlight(joints);
	}
  }
}

function drawThumbIndexHighlight(joints) {
  const THUMB = 4;
  const INDEX = 8;
  if (!joints[THUMB] || !joints[INDEX]) return;

  let t = joints[THUMB];
  let i = joints[INDEX];

  let tx = map(t.x, 0, 1, handGraphics.width, 0);
  let ty = map(t.y, 0, 1, 0, handGraphics.height);

  let ix = map(i.x, 0, 1, handGraphics.width, 0);
  let iy = map(i.y, 0, 1, 0, handGraphics.height);

  let d = dist(tx, ty, ix, iy);

  if (d < 30) {
    handGraphics.fill(128, 0, 128, 180);
    handGraphics.circle(tx, ty, 40);
  }
}

// ----------------------------
// 상단 Do/Re/Mi 표시
// ----------------------------
function drawTopLabel() {
  textAlign(CENTER, BOTTOM);
  textSize(36);
  fill(255);
  noStroke();
  text(predictedNote || "No Hand", width / 2, height / 2 + boxH / 2 - 20);
}