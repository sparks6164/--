import cv2, numpy as np, os, io
from PIL import Image
from rembg import remove
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image as MPImage, ImageFormat

BASE = os.getcwd()
OUT = os.path.join(BASE, "output_poses")
MODEL = r"C:\Users\Lenovo\Desktop\pose_landmarker.task"
os.makedirs(OUT, exist_ok=True)

PL = type('', (), {k:v for k,v in {
    'NOSE':0,'LEFT_SHOULDER':11,'RIGHT_SHOULDER':12,'LEFT_ELBOW':13,
    'RIGHT_ELBOW':14,'LEFT_WRIST':15,'RIGHT_WRIST':16,'LEFT_HIP':23,
    'RIGHT_HIP':24,'LEFT_KNEE':25,'RIGHT_KNEE':26,'LEFT_ANKLE':27,
    'RIGHT_ANKLE':28,'LEFT_EYE':2,'RIGHT_EYE':5,'LEFT_EAR':7,'RIGHT_EAR':8
}.items()})()

CONNS = [(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(9,10),(11,12),(11,23),
         (12,24),(23,24),(11,13),(13,15),(12,14),(14,16),(23,25),(25,27),
         (24,26),(26,28)]

def simg(p, img):
    if len(img.shape)==3 and img.shape[2]==3:
        r=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    elif len(img.shape)==3 and img.shape[2]==4:
        r=cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)
    else: r=img
    Image.fromarray(r).save(p)

def rmbg(p):
    with open(p,'rb') as f: d=f.read()
    return Image.open(io.BytesIO(remove(d)))

def ang(a,b,c):
    ba=np.array([a.x-b.x,a.y-b.y]); bc=np.array([c.x-b.x,c.y-b.y])
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6),-1,1)))

def pname(lm):
    n=lm[PL.NOSE]; ls=lm[PL.LEFT_SHOULDER]; rs=lm[PL.RIGHT_SHOULDER]
    le=lm[PL.LEFT_ELBOW]; re=lm[PL.RIGHT_ELBOW]; lw=lm[PL.LEFT_WRIST]; rw=lm[PL.RIGHT_WRIST]
    lh=lm[PL.LEFT_HIP]; rh=lm[PL.RIGHT_HIP]; lk=lm[PL.LEFT_KNEE]; rk=lm[PL.RIGHT_KNEE]
    la=lm[PL.LEFT_ANKLE]; ra=lm[PL.RIGHT_ANKLE]
    smy=(ls.y+rs.y)/2; hmy=(lh.y+rh.y)/2; kmy=(lk.y+rk.y)/2; amy=(la.y+ra.y)/2
    th=abs(hmy-smy); lh2=abs(amy-hmy); k2a=abs(amy-kmy)
    laa=ang(ls,le,lw); raa=ang(rs,re,rw)
    lua=lw.y<ls.y-0.02; rua=rw.y<rs.y-0.02; bu=lua and rua
    luh=lw.y<n.y; ruh=rw.y<n.y
    lhh=abs(lw.x-lh.x)<0.08 and abs(lw.y-lh.y)<0.06
    rhh=abs(rw.x-rh.x)<0.08 and abs(rw.y-rh.y)<0.06; hh=lhh or rhh
    lre=abs(lw.x-re.x)<0.1 and abs(lw.y-re.y)<0.07
    rle=abs(rw.x-le.x)<0.1 and abs(rw.y-le.y)<0.07; ac=lre or rle
    tl=th/(lh2+1e-6); sit=tl>1.5 or k2a<th*0.4
    st=abs(ls.y-rs.y); sw=abs(ls.x-rs.x)
    pts=[]
    pts.append("坐姿" if sit else "站立")
    if bu and (luh or ruh): pts.append("双手高举")
    elif bu: pts.append("双臂抬起")
    elif ac: pts.append("双臂交叉")
    elif hh: pts.append("叉腰")
    elif lua: pts.append("举左手")
    elif rua: pts.append("举右手")
    elif laa>150 and raa>150: pts.append("双臂下垂")
    elif laa<70 and raa<70: pts.append("屈臂")
    else: pts.append("手臂放松")
    if sw<0.08: pts.append("侧面")
    elif sw<0.14: pts.append("半侧面")
    else: pts.append("正面")
    return "_".join(pts)

def bbox(lm,w,h,pad=0.15):
    xs=[lm[i].x*w for i in range(33)]; ys=[lm[i].y*h for i in range(33)]
    bw=max(xs)-min(xs); bh=max(ys)-min(ys)
    return (max(0,int(min(xs)-bw*pad)),max(0,int(min(ys)-bh*pad)),
            min(w,int(max(xs)+bw*pad)),min(h,int(max(ys)+bh*pad)))

def ubbox(lm,w,h,pad=0.15):
    idxs=[0,2,5,7,8,11,12,13,14,15,16,23,24]
    xs=[lm[i].x*w for i in idxs]; ys=[lm[i].y*h for i in idxs]
    bw=max(xs)-min(xs); bh=max(ys)-min(ys)
    return (max(0,int(min(xs)-bw*pad)),max(0,int(min(ys)-bh*0.3)),
            min(w,int(max(xs)+bw*pad)),min(h,int(max(ys)+bh*0.3)))

def skel(img, lms, h, w):
    v=img.copy()
    for lm in lms:
        pts={}
        for i in range(33):
            lm_i=lm[i]; pts[i]=(int(lm_i.x*w),int(lm_i.y*h))
            if lm_i.visibility>0.3: cv2.circle(v,pts[i],3,(0,220,0),-1)
        for a,b in CONNS:
            if a in pts and b in pts:
                if lm[a].visibility>0.3 and lm[b].visibility>0.3:
                    cv2.line(v,pts[a],pts[b],(220,50,50),2)
    return v

def white(rgba, alpha):
    ph,pw=rgba.shape[:2]; wb=np.ones((ph,pw,3),dtype=np.uint8)*255
    af=alpha.astype(float)[:,:,np.newaxis]/255.0
    return (rgba[:,:,:3].astype(float)*af+wb.astype(float)*(1-af)).astype(np.uint8)

def proc(img_path, gid):
    print(f"\n{'='*50}\nProcessing: {os.path.basename(img_path)}")
    with open(img_path,'rb') as f: b=f.read()
    cvimg=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR)
    h,w=cvimg.shape[:2]
    rgba=np.array(rmbg(img_path)); alpha=rgba[:,:,3]
    opts=vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3, output_segmentation_masks=False)
    det=vision.PoseLandmarker.create_from_options(opts)
    rgb=cv2.cvtColor(cvimg,cv2.COLOR_BGR2RGB)
    res=det.detect(MPImage(image_format=ImageFormat.SRGB,data=rgb))
    det.close()
    rl=[]
    if res.pose_landmarks:
        pn=pname(res.pose_landmarks[0]); print(f"Pose: {pn}")
        x1,y1,x2,y2=bbox(res.pose_landmarks[0],w,h)
        c=rgba[y1:y2,x1:x2]; ca=alpha[y1:y2,x1:x2]
        p1=os.path.join(OUT,f"{gid:02d}_{pn}_全身像.jpg")
        simg(p1,cv2.cvtColor(white(c,ca),cv2.COLOR_RGB2BGR))
        print(f"  [1/3] {os.path.basename(p1)}"); rl.append(p1); gid+=1
        ux1,uy1,ux2,uy2=ubbox(res.pose_landmarks[0],w,h)
        uc=rgba[uy1:uy2,ux1:ux2]; ua=alpha[uy1:uy2,ux1:ux2]
        p2=os.path.join(OUT,f"{gid:02d}_{pn}_上半身像.jpg")
        simg(p2,cv2.cvtColor(white(uc,ua),cv2.COLOR_RGB2BGR))
        print(f"  [2/3] {os.path.basename(p2)}"); rl.append(p2); gid+=1
        ann=skel(rgb,res.pose_landmarks,h,w)
        ann_rgba=np.dstack([ann,alpha])
        ac=ann_rgba[y1:y2,x1:x2]; aca=alpha[y1:y2,x1:x2]
        p3=os.path.join(OUT,f"{gid:02d}_{pn}_骨架标注.jpg")
        simg(p3,cv2.cvtColor(white(ac,aca),cv2.COLOR_RGB2BGR))
        print(f"  [3/3] {os.path.basename(p3)}"); rl.append(p3); gid+=1
    else:
        wbg=np.ones((h,w,3),dtype=np.uint8)*255
        af=alpha.astype(float)[:,:,np.newaxis]/255.0
        seg=(rgba[:,:,:3].astype(float)*af+wbg.astype(float)*(1-af)).astype(np.uint8)
        for vn in ["全身像","上半身像","骨架标注"]:
            p=os.path.join(OUT,f"{gid:02d}_未识别_{vn}.jpg")
            simg(p,cv2.cvtColor(seg,cv2.COLOR_RGB2BGR))
            rl.append(p); gid+=1
    return rl,gid

def main():
    gid=1; ar=[]
    for i in ['q1.jpg','q2.jpg','q3.jpg']:
        p=os.path.join(BASE,i)
        if os.path.exists(p): r,gid=proc(p,gid); ar.extend(r)
    print(f"\nDone: {len(ar)} images in {OUT}")
    for i,p in enumerate(ar): print(f"  {i+1}. {os.path.basename(p)}")

if __name__=='__main__': main()
