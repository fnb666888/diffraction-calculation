"""Package offline rotating reconstructed-volume viewer and scientific exports."""
from pathlib import Path
import base64
import io
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage
from .pointcloud_tomography import make_figures,CASES

ROOT=Path(__file__).resolve().parents[1]


def run():
    out=ROOT/'outputs/pointcloud_holography';all_data=[];summary=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for name,(label,mode) in CASES.items():
        folder=out/name;c=json.loads((folder/'calibration.json').read_text(encoding='utf-8'))
        m=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        ref=np.load(folder/'input-reference.npz');result=np.load(folder/'reconstructed.npz')
        truth=ref['delta_n_volume'];volume=result['delta_n_volume'];points=ref['normalized_xyz'];phases=result['phase_projections_rad']
        print('Refreshing figures '+name,flush=True)
        # Reconstruct the reference projections solely for scientific comparison.
        # This is plotting only: independent reconstruction is already saved.
        from .pointcloud_tomography import project
        k=c['wave_number_per_m'];d=c['voxel_pitch_m']
        input_phases=np.asarray([project(truth,a)*k*d for a in c['angles_deg']])
        make_figures(folder,c,truth,volume,points,input_phases,phases)
        peak=float(volume.max());quantized=np.rint(np.clip(volume/peak,0,1)*255).astype(np.uint8)
        r,z,x=np.where(quantized>=3)
        sparse=np.column_stack([x,r,z,quantized[r,z,x]]).astype(np.uint8)
        original=np.rint(np.clip((points+.5)*95,0,95)).astype(np.uint8)
        holograms=[];phase_images=[];slm_amplitudes=[];slm_phases=[]
        packets=np.load(folder/'recordings.npz') if mode=='cgh' else None
        commands={key:packets[key] for key in ['slm_amplitude_commands','slm_phase_commands']} if packets is not None else {}
        for j in range(len(c['angles_deg'])):
            holograms.append('data:image/png;base64,'+base64.b64encode((folder/f'previews/hologram-{j:03d}.png').read_bytes()).decode())
            phase_images.append('data:image/png;base64,'+base64.b64encode((folder/f'previews/phase-{j:03d}.png').read_bytes()).decode())
            if packets is not None:
                for key,images in [('slm_amplitude_commands',slm_amplitudes),('slm_phase_commands',slm_phases)]:
                    buffer=io.BytesIO()
                    Image.fromarray(np.flipud(commands[key][j])).resize((256,256),Image.Resampling.BOX).save(buffer,format='PNG')
                    images.append('data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
        if packets is not None:packets.close()
        all_data.append({'name':name,'label':label,'mode':mode,'config':c,'metrics':m,'volume_n':96,
          'peak':peak,'volume':base64.b64encode(quantized.tobytes()).decode(),
          'recovered_voxels':base64.b64encode(sparse.tobytes()).decode(),
          'source_xyz':base64.b64encode(original.tobytes()).decode(),
          'holograms':holograms,'phase_images':phase_images,'slm_amplitudes':slm_amplitudes,'slm_phases':slm_phases})
        # Animation is an actual projection of the reconstructed volume, not the
        # original point cloud nor newly synthesized views at each video frame.
        frames=[]
        for angle in range(0,360,10):
            rotated=ndimage.rotate(np.maximum(volume,0),angle,axes=(1,2),reshape=False,order=1,prefilter=False)
            mip=rotated.max(axis=1);v=np.flipud(np.clip(mip/peak,0,1)**.35)
            gray=Image.fromarray(np.rint(v*255).astype(np.uint8)).resize((480,480),Image.Resampling.NEAREST).convert('RGB')
            canvas=Image.new('RGB',(480,545),'#101d30');canvas.paste(gray,(0,50));draw=ImageDraw.Draw(canvas)
            draw.text((14,12),f'{name}  数值重建体旋转 {angle}°',font=font,fill='white')
            frames.append(canvas)
        frames[0].save(folder/'reconstructed-rotation.gif',save_all=True,append_images=frames[1:],duration=100,loop=0)
        summary.append({'case':name,'label':label,'angles':len(c['angles_deg']),'metrics':m})
    (out/'viewer-data.js').write_text('window.HOLOGRAPHY_CASES='+json.dumps(all_data,ensure_ascii=False,separators=(',',':'))+';',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps({'cases':summary},ensure_ascii=False,indent=2),encoding='utf-8')
    template=(ROOT/'optics_sim/pointcloud_viewer.html').read_text(encoding='utf-8')
    # Embed data to keep the main deliverable independently usable without a server.
    script=(out/'viewer-data.js').read_text(encoding='utf-8')
    (out/'index.html').write_text(template.replace('/*__HOLOGRAPHY_DATA__*/',script),encoding='utf-8')
    print('Viewer packaged '+str(out/'index.html'),flush=True)


if __name__=='__main__':run()
