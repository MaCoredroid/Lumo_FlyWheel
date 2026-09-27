from pathlib import Path
import json,hashlib,re
R=Path('/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816')
S=R/'results/fr14_nvfp4_port_20260816/promotion_ab_eyeball.py'
source=S.read_bytes(); assert hashlib.sha256(source).hexdigest()=='f4cc87ade9f4e6866dafcdf8f2429085ede0dce5aceb96db380e0cc808aa2ea0'
n={'__name__':'read_only_eyeball_audit'};exec(compile(source,str(S),'exec'),n)
rows=[]
for run,arm,selected in [('fr14_promoab_Cqc10_20260824T074813Z','hydra27_fixed32_promoab_Cqc10',None),('fr14_promoab_Cqc16_20260819T222438Z','hydra27_fixed32_promoab_Cqc16','astropy__astropy-13236')]:
 root=R/'output'/run/arm/'swe_out/verified/per_task'
 for task in sorted(root.iterdir()):
  if not task.is_dir() or selected and task.name!=selected:continue
  trace=task/'qwen_trace.jsonl'; raw=trace.read_bytes();h=n['harvest'](trace);v=n['signatures']('\n'.join(h['texts']));thought='\n'.join(h['thinking'])
  rec={'n_tool_calls':len(h['tool_calls']),'thinking_chars':len(thought)}
  parse_errors=sum(1 for line in raw.decode(errors='replace').splitlines() if line.strip() and not valid_json(line)) if False else 0
  for line in raw.decode(errors='replace').splitlines():
   if line.strip():
    try:json.loads(line)
    except Exception:parse_errors+=1
  def pos(path,p):
   for line in path.read_text().splitlines():
    if line.startswith('vllm:spec_decode_num_accepted_tokens_per_pos_total{') and 'position="'+str(p)+'"' in line:return float(line.split()[-1])
   raise ValueError(str(path))
  pre,post=task/'vllm_metrics_pre.txt',task/'vllm_metrics_post.txt'
  c5=None
  if pre.exists() and post.exists():
   d4=pos(post,4)-pos(pre,4);d5=pos(post,5)-pos(pre,5);c5={'d4':d4,'d5':d5,'ratio':d5/d4 if d4 else None,'metric_pre_sha256':hashlib.sha256(pre.read_bytes()).hexdigest(),'metric_post_sha256':hashlib.sha256(post.read_bytes()).hexdigest()}
  meta=task/'runner_metadata.json';mbind=None
  if meta.exists():
   md=json.loads(meta.read_text());mbind=md.get('agent',{}).get('qwen_trace_capture',{});assert mbind.get('sha256')==hashlib.sha256(raw).hexdigest()
  rows.append({'run':run,'task':task.name,'trace_path':str(trace),'trace_sha256':hashlib.sha256(raw).hexdigest(),'trace_bytes':len(raw),'metadata_trace_binding':mbind,'trace_json_parse_errors':parse_errors,'assistant_records':h['assistant_turns'],'visible_chars':v['chars'],'visible_type_token_ratio':v['type_token_ratio'],'visible_tail_repeat_fraction':v['tail_repeat_fraction'],'visible_max_line_repeat':v['max_line_repeat'],'visible_most_repeated_8gram':v['longest_ngram_loop'],'thinking_chars':len(thought),'tool_calls':len(h['tool_calls']),'tool_names':sorted(set(str(x['name'])for x in h['tool_calls'])),'malformed_tool_calls':sum(x.get('json_parses')is False for x in h['tool_calls']),'flag':n['_degeneration_flag'](rec,v),'c5':c5,'visible_head':'\n'.join(h['texts'])[:1000],'visible_tail':'\n'.join(h['texts'])[-1600:]})
print(json.dumps({'schema':'cqc10-behavior-read-only-audit-v1','eyeball_source_path':str(S),'eyeball_source_sha256':hashlib.sha256(source).hexdigest(),'tasks':rows},indent=2))
