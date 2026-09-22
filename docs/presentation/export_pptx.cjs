#!/usr/bin/env node
/* Native editable PowerPoint export. All visible SVG content becomes text/shapes.
 * Run: node export_pptx.cjs --slides slides --deck deck.json --output file.pptx
 * PptxGenJS is loaded from the bundled workspace runtime or PPTXGENJS_MODULE.
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {spawnSync} = require('node:child_process');
const here = __dirname;
const options = Object.fromEntries(process.argv.slice(2).reduce((out, value, i, all) => {
  if (value.startsWith('--')) {
    if (!all[i + 1] || all[i + 1].startsWith('--')) throw new Error(`Missing value for ${value}`);
    out.push([value.slice(2), all[i + 1]]);
  }
  return out;
}, []));
const output = path.resolve(options.output || path.join(here, 'seif-architecture.pptx'));
const scenePath = path.resolve(options.scene || output.replace(/\.pptx$/i, '.scene.json'));
const reportPath = output.replace(/\.pptx$/i, '.pptx-qa.json');
const python = options.python || process.env.PYTHON || 'python3';
function pythonCall(args) {
  const result = spawnSync(python, [path.join(here, 'svg_to_scene.py'), ...args], {encoding:'utf8', maxBuffer:32*1024*1024});
  if (result.status !== 0) throw new Error(result.stderr || result.error || `Python exit ${result.status}`);
  return result.stdout;
}
if (!options.scene) {
  console.log(pythonCall(['--slides',path.resolve(options.slides || path.join(here,'slides')),
    '--deck',path.resolve(options.deck || path.join(here,'deck.json')), '--output',scenePath]).trim());
}
const scene = JSON.parse(fs.readFileSync(scenePath, 'utf8'));
const bundled = '/mnt/c/Users/doly2/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pptxgenjs';
const candidates = [process.env.PPTXGENJS_MODULE, 'pptxgenjs', bundled].filter(Boolean);
let PptxGenJS;
for (const candidate of candidates) {
  try { PptxGenJS = require(candidate); break; } catch (error) {
    if (candidate === candidates[candidates.length-1]) throw error;
  }
}
const pptx = new PptxGenJS();
pptx.defineLayout({name:'SEIF_WIDE', width:1600/120, height:900/120});
pptx.layout='SEIF_WIDE';
pptx.author='SEIF';
pptx.subject='Архитектура сервиса, оценка и измеренные эксперименты';
pptx.title='СЕЙФ · Архитектура и эксперименты';
pptx.company='SEIF';
pptx.lang='ru-RU';
pptx.theme={headFontFace:'Segoe UI',bodyFontFace:'Segoe UI',lang:'ru-RU'};
const inch = px => px/120;
const point = px => px*0.6;
function lineOptions(item) {
  if (!item.line) return {color:'111B17',transparency:100,width:0};
  const result={...item.line,width:point(item.lineWidth || 1),dashType:item.dash?'dash':'solid'};
  if(item.arrowEnd)result.endArrowType='arrow';
  if(item.arrowStart)result.beginArrowType='arrow';
  return result;
}
const warnings=[];
for (const [slideIndex, data] of scene.slides.entries()) {
  const slide=pptx.addSlide();
  slide.background={color:'111B17'};
  for (const [i,item] of data.items.entries()) {
    const name=`${String(slideIndex+1).padStart(2,'0')}-${item.kind}-${i}`;
    if (item.kind==='text') {
      if(!item.fill || item.fill.transparency===100)continue;
      slide.addText(item.text,{x:inch(item.x),y:inch(item.y),w:inch(item.w),h:inch(item.h),
        fontFace:item.fontFace,fontSize:point(item.fontSize),bold:item.bold,italic:item.italic,
        color:item.fill.color,transparency:item.fill.transparency,charSpacing:point(item.spacing || 0),
        align:item.align,valign:'top',margin:0,breakLine:false,wrap:false,fit:'none',
        paraSpaceAfter:0,paraSpaceBefore:0,lang:'ru-RU',objectName:name});
    } else if(item.kind==='line') {
      slide.addShape(pptx.ShapeType.line,{x:inch(Math.min(item.x1,item.x2)),y:inch(Math.min(item.y1,item.y2)),
        w:inch(Math.abs(item.x2-item.x1)),h:inch(Math.abs(item.y2-item.y1)),
        flipH:item.x2<item.x1,flipV:item.y2<item.y1,line:lineOptions(item),objectName:name});
    } else if(item.kind==='rect' || item.kind==='ellipse') {
      const shape=item.kind==='ellipse'?pptx.ShapeType.ellipse:item.radius?pptx.ShapeType.roundRect:pptx.ShapeType.rect;
      slide.addShape(shape,{x:inch(item.x),y:inch(item.y),w:inch(item.w),h:inch(item.h),
        rectRadius:item.radius?inch(item.radius):undefined,
        fill:item.fill || {color:'111B17',transparency:100},line:lineOptions(item),objectName:name});
    } else throw new Error(`Unexpected scene element ${item.kind}`);
  }
  const text=[data.title || `Slide ${slideIndex+1}`, '', data.notes || '', '', `Источники: ${data.source || 'SVG diagram source'}`,
    `SVG: ${data.file}`, `SVG SHA256: ${data.sha256}`, '',
    'Редактируемые native-элементы PowerPoint. Исходный SVG и offline HTML/PDF являются эталоном визуальной компоновки.'].join('\n');
  slide.addNotes(text);
  warnings.push(...(data.warnings || []).map(w=>({slide:slideIndex+1,...w})));
}
(async()=>{
  fs.mkdirSync(path.dirname(output),{recursive:true});
  await pptx.writeFile({fileName:output,compression:true});
  const structural=JSON.parse(pythonCall(['--validate-pptx',output]));
  const textIntegrityErrors=[];
  for (const [i,data] of scene.slides.entries()) {
    const text=data.items.filter(item=>item.kind==='text' && item.fill && item.fill.transparency!==100).map(item=>item.text).join('\0');
    const expected=crypto.createHash('sha256').update(text,'utf8').digest('hex');
    if (structural.slides[i]?.text_sha256!==expected) textIntegrityErrors.push(i+1);
  }
  const qa={output,conversion:scene.conversion,structural,text_integrity_errors:textIntegrityErrors,source_warnings:warnings};
  fs.writeFileSync(reportPath,JSON.stringify(qa,null,2));
  if(structural.slide_count!==scene.slides.length || structural.notes_count!==scene.slides.length || structural.picture_count!==0 || structural.canvas_bounds_errors.length || textIntegrityErrors.length) {
    throw new Error(`PPTX structural validation failed; see ${reportPath}`);
  }
  console.log(JSON.stringify({output,slides:structural.slide_count,editableTextRuns:structural.editable_text_runs,nativeShapes:structural.native_shapes,pictures:structural.picture_count,warnings:warnings.length,qa:reportPath},null,2));
})().catch(error=>{console.error(error.stack || error);process.exitCode=1;});
