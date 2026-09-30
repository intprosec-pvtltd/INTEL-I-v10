#!/usr/bin/env python3
"""Fresh DB only. Never regenerates existing Kubernetes secrets or local files."""
import argparse,base64,json,os,pathlib,secrets,subprocess,tempfile
p=argparse.ArgumentParser()
p.add_argument('--fresh-database',action='store_true',required=True)
a=p.parse_args()
root=pathlib.Path(__file__).resolve().parents[1];out=root/'secrets/intel-i-secrets.local.yaml'
result=subprocess.run(['kubectl','-n','intel-i','get','secret','intel-i-secrets','--ignore-not-found','-o','name'],check=True,capture_output=True,text=True)
if result.stdout.strip():raise SystemExit('Secret already exists. Reuse it; this script will not rotate encryption keys.')
if out.exists():raise SystemExit('Local secret file exists. Reuse it; refusing overwrite.')
rand=lambda:secrets.token_hex(32)
fernet=lambda:base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
password=rand();jwt=rand()
with tempfile.TemporaryDirectory() as td:
    key=pathlib.Path(td)/'jwt.pem'
    subprocess.run(['openssl','genpkey','-algorithm','RSA','-pkeyopt','rsa_keygen_bits:2048','-out',str(key)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    public=subprocess.check_output(['openssl','pkey','-in',str(key),'-pubout'],text=True)
    data={'POSTGRES_PASSWORD':password,'DATABASE_URL':f'postgresql://postgres:{password}@postgres:5432/cctv_db','JWT_PRIVATE_KEY':key.read_text(),'JWT_PUBLIC_KEY':public,'SECRET_KEY':jwt,'JWT_SECRET_KEY':jwt,'CAMERA_SOURCE_KEY':fernet(),'PERSON_EMBEDDING_KEY':fernet(),'MINIO_ACCESS_KEY':'inteli'+secrets.token_hex(8),'MINIO_SECRET_KEY':rand(),'ANPR_OCR_INTERNAL_TOKEN':rand(),'FRS_INTERNAL_TOKEN':rand(),'INFERENCE_INTERNAL_TOKEN':rand(),'INTEL_I_WORKER_PREVIEW_TOKEN':rand()}
obj={'apiVersion':'v1','kind':'Secret','metadata':{'name':'intel-i-secrets','namespace':'intel-i'},'type':'Opaque','stringData':data}
fd=os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f:json.dump(obj,f,indent=2)
print('Created secrets/intel-i-secrets.local.yaml (0600). Values were not printed. Back it up privately.')
