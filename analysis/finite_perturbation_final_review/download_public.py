"""Download only completed experiment public whitelist and verify every SHA."""
import hashlib
import io
import json
import pathlib
import subprocess
import tarfile

DEST = pathlib.Path('/Users/ruochengwang/ham/results/ising2d_physical_response_sequence_seed91_monitor/finite_perturbation')
REMOTE = r'''
import hashlib, io, json, pathlib, sys, tarfile
root=pathlib.Path('/mnt/nas-new/home/yange/ham/results/ising2d_positiveK_QCR_finite_perturbation_R5_seed20261017_vm225')
job=json.loads((root/'job.json').read_text())
assert job['workflow_complete'] and job['status'] in ('complete','scientific_stop')
assert len(job['stages'])==72 and all(s['returncode']==0 for s in job['stages'])
files={}
for name, expected in job['public_files'].items():
    rel=pathlib.PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and rel.suffix in ('.json','.png','.npz')
    source=root/name
    assert source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(root.resolve())
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    assert actual==expected, ('PUBLIC_SHA_MISMATCH',name)
    files[name]=(source,actual)
for source in [root/'job.json',root.parent.parent/'logs'/f'{root.name}.launch.json',root.parent.parent/'logs'/f'{root.name}.execution.json']:
    assert source.is_file() and not source.is_symlink()
    files[source.name]=(source,hashlib.sha256(source.read_bytes()).hexdigest())
manifest={'root':str(root),'status':job['status'],'workflow_complete':True,'stage_count':72,'all_returncodes_zero':True,'files':{n:{'sha256':h,'bytes':p.stat().st_size} for n,(p,h) in files.items()}}
data=json.dumps(manifest,ensure_ascii=False,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    item=tarfile.TarInfo('download_manifest.json');item.size=len(data);archive.addfile(item,io.BytesIO(data))
    for name,(source,_) in files.items():archive.add(str(source),arcname=name,recursive=False)
'''

def main():
    command = ['ssh','-F','/private/tmp/sjtu-vm.SiBJPz/config','-o','BatchMode=yes','-o','ConnectTimeout=15','225','cd /mnt/nas-new/home/yange/ham && .env/bin/python -']
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    process.stdin.write(REMOTE.encode());process.stdin.close()
    DEST.mkdir(parents=True,exist_ok=True)
    verified=[]
    with tarfile.open(fileobj=process.stdout,mode='r|') as archive:
        entries=iter(archive)
        first=next(entries)
        assert first.name=='download_manifest.json' and first.isfile()
        manifest_bytes=archive.extractfile(first).read()
        manifest=json.loads(manifest_bytes)
        expected=manifest['files']
        for member in entries:
            rel=pathlib.PurePosixPath(member.name)
            assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
            assert member.name in expected and rel.suffix in ('.json','.png','.npz')
            contents=archive.extractfile(member).read()
            record=expected[member.name]
            assert len(contents)==record['bytes'] and hashlib.sha256(contents).hexdigest()==record['sha256']
            destination=DEST/member.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(contents)
            verified.append(member.name)
    assert process.wait()==0 and set(verified)==set(expected)
    (DEST/'download_manifest.json').write_bytes(manifest_bytes)
    print(json.dumps({'downloaded_and_sha_verified':len(verified),'bytes':sum(v['bytes'] for v in expected.values()),'status':manifest['status'],'destination':str(DEST)},ensure_ascii=False))

if __name__=='__main__':main()
