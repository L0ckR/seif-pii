"""Build the offline SEIF deck from audited experiment metrics and vector scenes.

Run with Python 3.12+. Browser/PDF/PPTX export is a separate export.cjs step.
No service code, datasets, active credentials or remote assets are used.
"""
from __future__ import annotations

import html
import json
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
W, H = 1600, 900
C = dict(bg="#111b17", surface="#17231d", panel="#1c2a22", ink="#f0f0df",
         muted="#93a397", mint="#c0f0c5", orange="#eab291", border="#334138")
SANS = "Segoe UI, Arial, DejaVu Sans, sans-serif"
MONO = "Consolas, DejaVu Sans Mono, monospace"
SERIF = "Georgia, DejaVu Serif, serif"
slides = []
metrics = json.loads((HERE / "experiment-metrics.json").read_text())
experiments = {e["id"]: e for e in metrics["experiments"]}


def esc(value):
    return html.escape(str(value), quote=True)


class Slide:
    def __init__(self, title, chapter, source, notes=""):
        self.title, self.source, self.notes = title, source, notes
        self.elements = []
        self.rect(0, 0, W, H, C["bg"])
        self.line(64, 90, 1536, 90)
        for i in range(4):
            self.rect(65 + (i % 2) * 17, 34 + (i // 2) * 17, 12, 12,
                      C["mint"] if i == 1 else C["bg"], C["mint"], 1.8)
        self.text(111, 62, "СЕЙФ", 31, bold=True)
        self.text(286, 58, chapter.upper(), 16, C["muted"], font=MONO)
        self.text(1536, 58, "23.09.2026", 16, C["muted"], font=MONO, anchor="end")
        if title:
            self.text(64, 173, title, 58, maxw=1472)

    def rect(self, x, y, w, h, fill, stroke=None, sw=1, radius=0, opacity=1):
        self.elements.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}"'
                             f' stroke="{stroke or "none"}" stroke-width="{sw}" opacity="{opacity}"/>')

    def line(self, x1, y1, x2, y2, color=None, sw=1, dash=None):
        self.elements.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or C["border"]}"'
                             f' stroke-width="{sw}"' + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')

    def circle(self, x, y, r, fill, stroke=None, sw=1):
        self.elements.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def text(self, x, y, value, size=28, color=None, bold=False, font=SANS, italic=False, anchor="start", maxw=None):
        self.elements.append(f'<text x="{x}" y="{y}" font-size="{size}" font-family="{esc(font)}"'
                             f' font-weight="{600 if bold else 400}" font-style="{"italic" if italic else "normal"}"'
                             f' text-anchor="{anchor}" fill="{color or C["ink"]}"'
                             + (f' data-max-width="{maxw}"' if maxw else '') + f'>{esc(value)}</text>')

    def lines(self, x, y, values, size=28, color=None, gap=None, **kwargs):
        for i, value in enumerate(values):
            self.text(x, y + i * (gap or size * 1.38), value, size, color, **kwargs)

    def pill(self, x, y, label, color=None, width=None):
        width = width or len(label) * 10.3 + 34
        color = color or C["mint"]
        self.rect(x, y, width, 36, C["surface"], color)
        self.text(x + 15, y + 25, label, 16, color, font=MONO)

    def card(self, x, y, w, h, number, title, body, accent=None):
        accent = accent or C["mint"]
        self.rect(x, y, w, h, C["surface"], C["border"])
        self.text(x + 28, y + 44, number, 17, accent, font=MONO)
        self.lines(x + 28, y + 105, title if isinstance(title, list) else [title], 34, gap=42, maxw=w-56)
        offset = 208 if isinstance(title, list) and len(title) > 1 else 158
        self.lines(x + 28, y + offset, body, 25, C["muted"], gap=36, maxw=w-56)

    def finish(self):
        i = len(slides) + 1
        self.line(64, 830, 1536, 830)
        self.text(64, 866, self.source, 15, C["muted"], maxw=1350)
        self.text(1536, 866, f"{i:02d} / 16", 16, C["muted"], font=MONO, anchor="end")
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900" role="img" aria-label="{esc(self.title)}">'
        svg += ''.join(self.elements) + '</svg>'
        slides.append(dict(title=self.title, svg=svg, notes=self.notes, source=self.source))


def diagram(file_name, title, source, notes):
    svg = (HERE / "diagrams" / file_name).read_text()
    # Keep agent-authored vectors, with the same export footer as the other slides.
    overlay = Slide("", "", "")
    overlay.elements = []
    overlay.rect(0, 830, W, 70, C["bg"])
    overlay.line(64, 830, 1536, 830)
    overlay.text(64, 866, source, 15, C["muted"], maxw=1350)
    overlay.text(1536, 866, f"{len(slides)+1:02d} / 16", 16, C["muted"], font=MONO, anchor="end")
    svg = svg.replace('</svg>', ''.join(overlay.elements) + '</svg>')
    slides.append(dict(title=title, svg=svg, notes=notes, source=source))


def percentage(value):
    return f"{value:.2f}".replace('.', ',')


def build():
    s = Slide("", "Архитектура · эксперименты · оценка", "Срез исходников: 15eea67 · experiment/rubert-tensorrt · существующий web-стиль", "Архитектура сервиса и экспериментального GPU-профиля. Числа взяты из сохранённых отчётов; новые нагрузочные или модельные эксперименты для этой презентации не запускались.")
    s.title = "Данные работают. Личное остаётся личным."
    s.pill(64, 160, "МОДУЛЬ ЗАЩИТЫ ПЕРСОНАЛЬНЫХ ДАННЫХ", width=439)
    s.lines(64, 302, ["Данные работают."], 82, gap=94, maxw=900)
    s.lines(64, 406, ["Личное остаётся", "личным."], 83, C["mint"], gap=97, font=SERIF, italic=True, maxw=900)
    s.lines(68, 628, ["Локальная защита перед LLM.", "Обратимость. Контекст. Проверяемые результаты."], 28, C["muted"], gap=43)
    s.rect(1030, 182, 506, 548, C["surface"], C["border"])
    for x in range(1052, 1530, 28):
        for y in range(206, 720, 28):
            s.circle(x, y, 1, C["border"])
    s.text(1060, 221, "PRIVATE DATA FLOW", 16, C["muted"], font=MONO)
    for x, label in [(1056,"PERSON"),(1220,"PASSPORT"),(1384,"EMAIL")]:
        s.rect(x, 280, 130, 42, C["panel"], C["border"])
        s.text(x+65, 307, label, 17, C["orange"], font=MONO, anchor="middle")
        s.line(x+65,322,x+65,368,C["border"],2)
        s.line(x+65,368,1280,368,C["border"],2)
    s.line(1280,368,1280,399,C["mint"],2)
    s.rect(1180, 399, 208, 167, "#304b36", "#6f9271", 2)
    s.rect(1193, 412, 182, 141, "#253c2b", "#506e51", 2)
    s.circle(1284, 480, 43, C["panel"], C["mint"], 2)
    for x,y,x2,y2 in [(1255,480,1313,480),(1284,451,1284,509),(1263,459,1305,501)]:
        s.line(x,y,x2,y2,C["mint"],3)
    s.line(1284,566,1284,624,C["mint"],2)
    s.rect(1124, 624, 320, 45, C["bg"], C["mint"])
    s.text(1284,654,"⟦PD:PERSON:…⟧",23,C["mint"],font=MONO,anchor="middle")
    s.finish()

    s = Slide("Удачная основа. Чёткие границы роста.", "01 / Архитектурная оценка", "Оценка по исходникам · подробные доказательства: architecture-assessment.md", "Это качественная архитектурная оценка, не численный рейтинг и не подтверждение production SLA. Критерии хакатона оценивают демонстрационный прокси, а не завершённую production-платформу.")
    s.lines(64,232,["Сильный демо-прокси; для пилота важнее закрыть измеримые пробелы, чем добавлять сервисы."],27,C["muted"],maxw=1472)
    s.card(64,292,468,375,"01 / РАЗДЕЛЕНИЕ","Логика по границам",["API и правила — отдельно от NER.","Хранилище — вне модели.","LLM вызывает потребитель.","Профили можно менять явно."])
    s.card(554,292,468,375,"02 / КОНТРОЛЬ","Прозрачные решения",["Политика на потребителя.","Причины и Unicode-диапазоны.","Ошибка NER не скрывается.","Восстановление проверяется."])
    s.card(1044,292,492,375,"03 / ОГРАНИЧЕНИЯ","Нужен следующий шаг",["NER ограничивает GPU-профиль.","Границы сущностей неточны.","Golden уже изучен разработкой.","HA задана конфигурацией," ,"но не доказывает zero-RPO."],C["orange"])
    s.text(64,755,"ВЕРДИКТ",17,C["mint"],font=MONO)
    s.text(232,761,"Развивать текущую модульную схему и измерять каждое изменение.",31,maxw=1300)
    s.finish()

    diagram("01-logical-architecture.svg", "Одна граница защиты — несколько независимых компонентов", "Источники: seif/app.py · seif/ner.py · seif/vault.py · scripts/client_example.py", "Схема нового payload_id. При повторе запрос может обслуживаться из зашифрованного vault без повторного NER. Внешний LLM не вызывается сервером СЕЙФ: его оркестрирует потребитель. Это граница маршрутизации, а не гарантия безошибочного распознавания.")
    diagram("02-reversible-token-flow.svg", "Защитить, обработать, восстановить", "Источники: seif/transform.py · seif/vault.py · seif/app.py", "Token допускает повторение и перестановку известных токенов в новом окружении. Mask/synthetic требуют точный защищённый результат. Восстановление ограничено tenant, payload_id, политикой и временем жизни записи.")
    diagram("04-detection-pipeline.svg", "От исходного текста до проверенных замен", "Источники: seif/detector.py · seif/rubert_decoder.py · seif/ner_contract.py", "Сохраняются исходные Unicode code-point offsets. 21 native-категория RuBERT отображается в 14 транспортных типов; это не количество всех типов правил. Ошибка настроенного NER завершает запрос ошибкой, а не успешным неполным результатом.")
    diagram("03-deployment-profiles.svg", "Развёртывание без смешения профилей", "Источники: compose.yaml · compose.ner.yaml · deploy/k8s · deploy/service/README.md", "Базовый Compose, optional CPU NER и отдельный экспериментальный GPU-процесс не являются одним одинаково измеренным стендом. K8s-манифесты подготовлены; наличие yaml не доказывает работающий production-кластер. GPU-образ и GPU-оркестрация отдельно не реализованы.")

    s = Slide("Политика задаёт смысл защиты.", "02 / Безопасность и расширяемость", "Источники: config/policies.yaml · seif/config.py · seif/app.py · seif/vault.py", "Сервис поддерживает allowlist потребителей, ключи API, перечни типов, комбинации типов, custom rules и отдельное разрешение восстановления. Capture исходных текстов — самостоятельная опция: его файлы не защищены шифрованием и TTL vault автоматически.")
    s.card(64,255,708,235,"01 / ДОСТУП","Система-потребитель",["Ключи, allowlist и отдельное разрешение unmask.","Демо-режим отделён от ограниченного режима."])
    s.card(794,255,742,235,"02 / ПРАВИЛА","Настройка без правки ядра",["Типы ПД, комбинации и custom rules.","Публичный контекст не равен личным данным."])
    s.card(64,514,708,255,"03 / ДАННЫЕ","Шифрованное соответствие",["AES-GCM и TTL для записей восстановления.","Namespace потребителя + payload_id.","Общий мастер-ключ нужен всем репликам."])
    s.card(794,514,742,255,"04 / НАБЛЮДАЕМОСТЬ","Метрики без исходного текста",["Этапы, длительность, типы и ошибки.","Захват raw-запросов — отдельная опция:", "нужны свой retention и контроль доступа."],C["orange"])
    s.finish()

    s = Slide("Кандидаты проверяли на трёх корпусах.", "03 / Основные эксперименты", "Источники и условия каждой ячейки: experiment-metrics.json · единицы: F1 маскирования, %", "Это сравнение сервисных конфигураций, а не чистый рейтинг моделей: контракты и версии правил различаются. spaCy и GLiNER пересчитаны на текущих правилах по сохранённым предсказаниям. GLiNER обученному декодеру передавали person/location и конкурирующий organization; другие PII-классы не запрашивались. Строгая проверка предобработки 5 095 текстов не выявила ошибки выравнивания. LFM указан с address-proxy; его native hybrid с 40 типами показан отдельно. Organizer размечен ИИ, а все корпуса уже изучались в разработке.")
    s.text(64,228,"Текущие правила: spaCy, GLiNER, новый RuBERT. LFM и прежний RuBERT — исторические профили.",25,C["muted"])
    xs=[86,720,940,1150,1388]
    for x,label in zip(xs,["СЕРВИСНЫЙ ПРОФИЛЬ","ORGANIZER","PII-BENCH","REDMADROBOT","КОНТРАКТ"]):
        s.text(x,292,label,17,C["muted"],font=MONO,anchor="start" if x==86 else "middle")
    labels=["spaCy ru_core_news_sm", "GLiNER 2.5 / PyTorch", "GLiNER 2.5 / ONNX", "LFM2.5 / address-proxy", "RuBERT / прежний сервис", "RuBERT / текущий сервис"]
    for i,(e,label) in enumerate(zip(metrics["experiments"][:6],labels)):
        y=315+i*68
        s.rect(64,y,1472,62,C["panel"] if i==5 else C["surface"])
        s.text(86,y+41,label,27,C["mint"] if i==5 else C["ink"])
        for x,k in zip(xs[1:4],['organizer','pii','redmadrobot']):
            s.text(x,y+41,percentage(e['mask_f1'][k]['value']),29,C["mint"] if i==5 else C["ink"],font=MONO,anchor="middle")
        s.text(1388,y+40,'14 типов' if i==5 else 'PERSON / LOC',17,C["muted"],font=MONO,anchor="middle")
    lfm_native=experiments['lfm_hybrid']['native_mask_f1']['official_hybrid']['pii']['value']
    s.text(64,751,f"LFM native hybrid: PII-Bench {percentage(lfm_native)}% — 40 типов, без правил СЕЙФ; отдельный профиль.",21,C["mint"],maxw=1472)
    s.text(64,780,"GLiNER пересчитан по сохранённым предсказаниям. Полный набор PII-классов требует нового inference.",20,C["orange"],maxw=1472)
    s.text(64,809,"446 / 1 810 / 2 839 текстов. Разные контракты и версии правил; это не рейтинг одних только моделей.",19,C["muted"],maxw=1472)
    s.finish()

    s = Slide("Обновление RuBERT улучшило маскирование.", "04 / Контролируемое сравнение", "Источник: rubert-upgrade/quality-v5.json · одинаковые тексты, gold и scoring для трёх систем", "F1 считается по закрытым буквам и цифрам независимо от типа и точных границ. pii-guard — pinned raw-text ablation: правила, word-decoder и merge, без normalize/translit/base64. Это не воспроизведение его опубликованного model-card score.")
    s.text(64,227,"Предыдущий RuBERT → текущий RuBERT → контроль pii-guard на исходном тексте.",26,C["muted"])
    old=[96.8791,74.0731,69.7101]; new=[97.1285,77.8639,93.9041]; guard=[81.8104,78.5479,93.6748]
    for i,(name,n) in enumerate([('Organizer','446 · AI silver'),('PII-Bench','1 810'),('RedMadRobot','2 839')]):
        x=64+i*498
        s.rect(x,279,476,450,C["surface"],C["border"])
        s.text(x+25,329,name,34)
        s.text(x+25,365,n,18,C["muted"],font=MONO)
        for j,(label,val,color) in enumerate([('Раньше',old[i],C['muted']),('Сейчас',new[i],C['mint']),('pii-guard*',guard[i],C['orange'])]):
            yy=418+j*91
            s.text(x+25,yy,label,22,color)
            s.text(x+444,yy,percentage(val)+'%',26,color,font=MONO,anchor='end')
            s.rect(x+25,yy+18,420,16,C['bg'])
            s.rect(x+25,yy+18,420*val/100,16,color)
        s.text(x+25,701,'Шкала: 0–100%',16,C['muted'],font=MONO)
    s.text(64,784,"* Правила + модель + merge без полного препроцессинга. Organizer — предварительная AI-разметка.",21,C["muted"])
    s.finish()

    s = Slide("Скрыть символы ≠ точно выделить сущность.", "05 / Качество без подмены метрик", "Источник: quality-v5.json · RedMadRobot · common14 исключает 19 предсказаний СЕЙФ / 253 guard", "Mask F1 выше raw-text pii-guard на RMR, но recall ниже: 90,8026% против 92,7205%. Exact common14 остаётся ниже: 86,7601% против 89,8458%. Common14 исключает внеобластные предсказания: 19 у СЕЙФ и 253 у guard. Без этих исключений exact F1 равен 86,6113% и 87,8188%. Из 801 strict FN: 386 — правильный тип, но другие границы; 60 — другой тип; 91 — частичное покрытие; 264 — не защищены. На organizer и PII exact-span снизился из-за разделения на компоненты. Сохранение маски не отменяет эту регрессию.")
    s.text(64,229,'RedMadRobot · * pii-guard: правила + модель + merge без полного препроцессинга.',24,C['muted'])
    for x,label,val,ref,color in [(64,'F1 МАСКИРОВАНИЯ','93,90%','pii-guard*: 93,67%',C['mint']),(818,'EXACT F1 · COMMON14','86,76%','pii-guard*: 89,85%',C['orange'])]:
        s.rect(x,263,718,272,C['surface'],C['border'])
        s.text(x+30,308,label,20,C['muted'],font=MONO)
        s.text(x+25,426,val,112,color)
        s.text(x+30,493,ref,27,C['muted'])
    s.text(64,602,"Оставшиеся 801 несовпадение exact-span",34)
    for i,(value,lines) in enumerate([('386',['Иные границы,','нужные символы скрыты']),('60',['Другая категория,','символы скрыты']),('91',['Защищены','частично']),('264',['Размеченные сущности','не защищены'])]):
        x=64+i*375
        s.text(x,686,value,60,C['mint'] if i<2 else C['orange'])
        s.lines(x,735,lines,23,C['muted'],gap=31)
    s.finish()

    s = Slide("Скорость измеряли полным HTTP-циклом.", "06 / Нагрузка и ограничения замера", "Источники: GLiNER HTTP C4 · RuBERT http-baseline-control / http-upgraded-final C8", "Группы имеют разные условия и не доказывают кратного ускорения от одной смены модели. GLiNER: 446 запросов маскирования, C4; RuBERT: 22 300, C8. Один API-worker, один NER-worker, память вместо Redis, loopback, реальная модель, capture отключён. GLiNER RPS относится к историческому сервису: после нового CPU replay нагрузка не запускалась. Эти результаты не подтверждают SLA 1 000 RPS для GPU-профиля. У LFM HTTP не измерялся; скорость модели в texts/s нельзя выдавать за HTTP RPS.")
    s.text(64,228,"Два протокола — сравнения внутри каждой пары. Все значения ниже — успешные mask RPS.",25,C['muted'])
    for x,tag,subtitle,values,maxval in [(64,'GLiNER 2.5','Исторический сервис · 446 запросов · C4',[('PyTorch',31.89),('ONNX',49.86)],60),(818,'RuBERT TensorRT','22 300 запросов · C8 · 50 циклов',[('Прежний',528.86),('Текущий',533.74)],600)]:
        s.rect(x,273,718,393,C['surface'],C['border'])
        s.text(x+30,324,tag,35)
        s.text(x+30,366,subtitle,22,C['muted'])
        for j,(label,value) in enumerate(values):
            yy=438+j*112
            s.text(x+30,yy,label,28)
            s.text(x+678,yy,percentage(value),40,C['mint'],font=MONO,anchor='end')
            s.rect(x+30,yy+25,646,22,C['bg'])
            s.rect(x+30,yy+25,646*value/maxval,22,C['mint'] if j else '#657b67')
        s.text(x+30,639,f'Шкала: 0–{maxval} RPS',16,C['muted'],font=MONO)
    for x,val,label in [(64,'19,58 мс','p95 текущего RuBERT'),(590,'0 / 22 300','ошибок HTTP / запросов'),(1160,'+0,92%','наблюдаемая разница RPS')]:
        s.text(x,730,val,42,C['mint'])
        s.text(x,770,label,22,C['muted'])
    s.finish()

    s = Slide("Четыре риска, которые стоит закрыть.", "07 / Архитектурный разбор", "Оценка кода, не результаты penetration/failover-теста · architecture-assessment.md", "Это предложения, а не изменения сервиса в рамках презентации. Тело запроса ограничено по размеру и времени, но авторизация API выполняется после чтения и разбора JSON. В одном NER-процессе модель исполняется последовательно; CPU-HPA API не решает перегрузку GPU. Асинхронная репликация Redis требует согласования допустимой потери последних записей. Raw capture хранится отдельно от шифрования и TTL vault.")
    cards=[('P1 / РЕСУРСЫ','Допуск до затрат',['Раннее отклонение чужих запросов.','Лимиты до чтения и разбора тела.']),('P1 / NER','Очередь по модели',['Ограниченная очередь и backpressure.','Масштабирование по очереди/latency.']),('P1 / ДАННЫЕ','RPO и срок жизни',['Проверить failover и повторы запросов.','Определить допустимую потерю записей.']),('P1 / CAPTURE','Отдельное хранение raw',['Шифрование, retention и права доступа.','Убедиться, что capture отключён в пилоте.'])]
    for i,(tag,title,body) in enumerate(cards):
        s.card(64+(i%2)*754,258+(i//2)*265,718,239,tag,title,body,C['orange'])
    s.finish()

    s = Slide("Следующий шаг — измеримый пилот.", "08 / План развития", "Предложения по развитию · критерии приёмки нужно согласовать с владельцем пилота", "Приоритет — независимый holdout и совместная проверка маскирования, точных границ и типов. Следующий корректный эксперимент GLiNER требует одинакового с RuBERT перечня PII-классов и нового запуска модели. Batching и реплики нужно оценивать по очереди, latency и смешанной нагрузке. До пилота отдельно проверяются отказ компонентов, восстановление и ротация ключей.")
    s.card(64,275,468,414,'01 / КАЧЕСТВО',['Независимая','проверка'],['Holdout с ручной разметкой.','Mask F1, exact F1, recall типов.','Отдельная проверка РФ и СНГ.','Сборка ФИО для synthetic.'])
    s.card(554,275,468,414,'02 / СКОРОСТЬ',['GPU-профиль','под нагрузкой'],['Длинные тексты, mixed workload.','Очередь и batching — абляции.','p95 / p99 и доля ошибок.','Повторные прогоны без кэша.'])
    s.card(1044,275,492,414,'03 / ПИЛОТ',['Рабочий контур','и эксплуатация'],['Секреты, ingress/TLS и права.', 'Retention capture и rotation ключей.', 'Восстановление после сбоев.', 'SLO и владелец реакции на ошибки.'])
    s.text(64,763,'Критерий успеха: качество, обратимость и latency сохраняются на новых данных и при отказах.',25,C['mint'],maxw=1472)
    s.finish()

    s = Slide("Демо на две минуты: видно каждое решение.", "09 / Сценарий защиты", "Вымышленный пример · token ниже схематический · API: /v1/mask → /v1/unmask", "Сценарий отражает критерии хакатона: контекст, маскирование, восстановление, настройка политики и метрики. Используйте вымышленные данные, текущий профиль и фактический ответ API. Токен на слайде схематический: модель может выделить несколько компонентов ФИО. Не выдавайте экспериментальный GPU-профиль за уже развёрнутый публичный сервис.")
    steps=[('01','ПОДАТЬ ТЕКСТ','Личное рядом с публичным'),('02','ЗАЩИТИТЬ','Показать найденные типы'),('03','ВОССТАНОВИТЬ','Сверить точное совпадение'),('04','ИЗМЕНИТЬ ПОЛИТИКУ','Показать управляемость')]
    for i,(n,label,desc) in enumerate(steps):
        x=64+i*376
        s.text(x,285,n,44,C['mint'],font=MONO)
        s.text(x,335,label,18,C['muted'],font=MONO)
        s.lines(x,382,textwrap.wrap(desc,24),26,gap=34)
    s.rect(64,471,718,242,C['surface'],C['border'])
    s.rect(818,471,718,242,C['surface'],C['border'])
    s.text(94,516,'ИСХОДНЫЙ ТЕКСТ',17,C['muted'],font=MONO)
    s.lines(94,574,['Поэт Александр Пушкин.', 'Клиент: Иванов Иван Иванович.'],30,gap=50)
    s.text(848,516,'ИЗМЕНИЛАСЬ ТОЛЬКО ЛИЧНАЯ ЧАСТЬ',17,C['muted'],font=MONO)
    s.text(848,574,'Поэт Александр Пушкин.',30)
    s.text(848,624,'Клиент: ⟦PD:PERSON:…⟧',29,C['mint'],font=MONO)
    s.text(64,780,'На экране: реальные типы, длительность, режим и результат восстановления.',26,C['muted'])
    s.finish()

    s = Slide("Сильнее становится вся цепочка.", "10 / Вывод", "СЕЙФ · архитектурная оценка и проверенные эксперименты · 23 сентября 2026", "Рекомендация — сохранить текущие модульные границы и сосредоточиться на объединении сущностей, точных границах и независимом holdout. Полного превосходства над pii-guard нет. 533,74 RPS — локальное измерение конкретного GPU-профиля. 3 119 — ранее прошедшие основные тесты; ещё 161 проверка выполнена в модельном окружении, а 35 основных тестов пропущены из-за опциональных зависимостей и локальных сервисов.")
    s.lines(64,301,['Управляемая защита.', 'Обратимый результат.', 'Проверяемый прогресс.'],66,C['mint'],gap=91,font=SERIF,italic=True,maxw=1010)
    s.rect(1070,252,466,479,C['surface'],C['border'])
    for yy,num,label in [(320,'5 095','текстов в общем сравнении'),(458,'533,74','RPS в финальном GPU-прогоне'),(596,'3 119','прошедших основных тестов')]:
        s.text(1100,yy,num,63,C['mint'])
        s.text(1100,yy+42,label,22,C['muted'],maxw=410)
    s.text(64,647,'Фокус следующей итерации',19,C['muted'],font=MONO)
    s.lines(64,701,['Точные границы и типы на новых данных.', 'Скорость и восстановление при реальных ограничениях.'],30,gap=44,maxw=970)
    s.finish()

    s = Slide("Как проверять цифры этой презентации.", "Приложение / Методология и источники", "Полные ссылки, JSON pointers и SHA256: experiment-metrics.json · architecture-evidence.json", "Архитектура проверена по коммиту 15eea67. Для чисел сохранены пути отчётов, JSON pointers и SHA256. Повторная проверка GLiNER сохраняет тот же состав 5 095 текстов и версии runtime. PDF критериев хакатона использованы как контекст презентации. Исходные тексты датасетов и секреты в материалы не переносились.")
    s.card(64,258,708,249,'01 / ВЫБОРКА','5 095 фиксированных текстов',['446 organizer: 329 certain + 117 uncertain.','1 810 PII-Bench; 2 839 RedMadRobot.','Два старых исключения выравнивания RMR.'])
    s.card(794,258,742,249,'02 / МЕТРИКИ','Сравниваем одинаковое с одинаковым',['Mask F1 — закрытые буквы и цифры.','Exact F1 — тип и исходные границы.','Model texts/s не равны HTTP RPS.'])
    s.lines(64,571,['Ключевые отчёты'],30)
    s.lines(64,620,['rubert-upgrade/quality-v5.json','rubert-upgrade/http-upgraded-final.json','gliner25-onnx/comparison-gpu.json','lfm25-pii/comparison.json'],21,C['mint'],font=MONO,gap=34)
    s.lines(850,573,['Условия вывода'],30)
    s.lines(850,620,['Organizer — AI silver, не официальный эталон.','Корпуса уже изучались: это не blind holdout.','pii-guard — raw-text ablation, не весь pipeline.','Kubernetes-конфигурация ≠ проверенный SLA.'],24,C['orange'],gap=39,maxw=680)
    s.finish()


def main():
    build()
    assert len(slides) == 16
    for i, slide in enumerate(slides, 1):
        for identity in re.findall(r'\bid="([^"]+)"', slide['svg']):
            prefixed=f'slide{i}-{identity}'
            slide['svg']=slide['svg'].replace(f'id="{identity}"', f'id="{prefixed}"').replace(f'url(#{identity})',f'url(#{prefixed})')
        slide['svg']=re.sub(r'aria-labelledby="([^"]+)"', lambda m:'aria-labelledby="'+' '.join(f'slide{i}-{v}' for v in m[1].split())+'"', slide['svg'])
        ET.fromstring(slide['svg'])
        (HERE/'slides'/f'{i:02d}.svg').write_text(slide['svg'])
    (HERE/'deck.json').write_text(json.dumps([{k:v for k,v in s.items() if k != 'svg'} for s in slides],ensure_ascii=False,indent=2))
    template=(HERE/'template.html').read_text()
    scene='\n'.join(f'<section class="slide" data-index="{i}" aria-label="{esc(s["title"])}">{s["svg"]}</section>' for i,s in enumerate(slides))
    nav=''.join(f'<option value="{i}">{i+1:02d} · {esc(s["title"])}</option>' for i,s in enumerate(slides))
    notes=json.dumps([s['notes'] for s in slides],ensure_ascii=False).replace('</','<\\/')
    rendered=template.replace('@@SLIDES@@',scene).replace('@@OPTIONS@@',nav).replace('@@NOTES@@',notes)
    (HERE/'index.html').write_text(rendered)
    print(f'Built {len(slides)} slides, offline HTML and editable vector SVG sources.')


if __name__=='__main__':
    main()
