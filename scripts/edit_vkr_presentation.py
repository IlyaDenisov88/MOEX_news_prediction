from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
import shutil

WORK = Path("/Users/krokodile/Desktop/moex_news_app copy/outputs/manual-20260529-122854/presentations/vkr-defense-edit")
SRC = WORK / "ВКР_Денисов_Презентация_source.pptx"
UNPACKED = WORK / "edited_unpacked"
OUT = WORK / "output" / "ВКР_Денисов_Презентация_доработанная.pptx"

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"p": P, "a": A, "r": R}
EMU = 914400


def qn(ns, tag):
    return f"{{{ns}}}{tag}"


def emu(inches):
    return str(int(round(inches * EMU)))


def parse(slide):
    return etree.parse(str(UNPACKED / f"ppt/slides/slide{slide}.xml"))


def save(tree, slide):
    tree.write(str(UNPACKED / f"ppt/slides/slide{slide}.xml"), xml_declaration=True, encoding="UTF-8", standalone=True)


def max_shape_id(tree):
    ids = []
    for el in tree.xpath("//p:cNvPr/@id", namespaces=NS):
        try:
            ids.append(int(el))
        except ValueError:
            pass
    return max(ids or [1])


def sp_tree(tree):
    return tree.xpath("//p:cSld/p:spTree", namespaces=NS)[0]


def insert_before_ext(tree, element):
    st = sp_tree(tree)
    ext_lst = st.find(qn(P, "extLst"))
    if ext_lst is not None:
        st.insert(list(st).index(ext_lst), element)
    else:
        st.append(element)


def clear_shape_text_by_id(tree, shape_id):
    sp = tree.xpath(f'//p:sp[p:nvSpPr/p:cNvPr/@id="{shape_id}"]', namespaces=NS)[0]
    tx_body = sp.find(qn(P, "txBody"))
    if tx_body is not None:
        for p_el in tx_body.findall(qn(A, "p")):
            tx_body.remove(p_el)
    return sp


def add_paragraph(tx_body, text, size=13, bold=False, color="222222", bullet=False, space_after=350):
    p_el = etree.SubElement(tx_body, qn(A, "p"))
    ppr = etree.SubElement(p_el, qn(A, "pPr"))
    if bullet:
        etree.SubElement(ppr, qn(A, "buChar"), char="•")
    else:
        etree.SubElement(ppr, qn(A, "buNone"))
    ppr.set("marL", "260000" if bullet else "0")
    ppr.set("indent", "-160000" if bullet else "0")
    ppr.set("spcAft", str(space_after))
    r_el = etree.SubElement(p_el, qn(A, "r"))
    rpr = etree.SubElement(r_el, qn(A, "rPr"), lang="ru-RU", sz=str(size * 100))
    if bold:
        rpr.set("b", "1")
    solid = etree.SubElement(rpr, qn(A, "solidFill"))
    etree.SubElement(solid, qn(A, "srgbClr"), val=color)
    etree.SubElement(rpr, qn(A, "latin"), typeface="Arial")
    etree.SubElement(rpr, qn(A, "cs"), typeface="Arial")
    etree.SubElement(r_el, qn(A, "t")).text = text
    return p_el


def add_textbox(tree, sid, name, x, y, w, h, paragraphs, font_size=13, color="222222",
                fill=None, line=None, radius=False, margin=0.08):
    sp = etree.Element(qn(P, "sp"))
    nv = etree.SubElement(sp, qn(P, "nvSpPr"))
    etree.SubElement(nv, qn(P, "cNvPr"), id=str(sid), name=name)
    etree.SubElement(nv, qn(P, "cNvSpPr"), txBox="1")
    etree.SubElement(nv, qn(P, "nvPr"))
    sppr = etree.SubElement(sp, qn(P, "spPr"))
    xfrm = etree.SubElement(sppr, qn(A, "xfrm"))
    etree.SubElement(xfrm, qn(A, "off"), x=emu(x), y=emu(y))
    etree.SubElement(xfrm, qn(A, "ext"), cx=emu(w), cy=emu(h))
    etree.SubElement(sppr, qn(A, "prstGeom"), prst="roundRect" if radius else "rect").append(etree.Element(qn(A, "avLst")))
    if fill:
        sf = etree.SubElement(sppr, qn(A, "solidFill"))
        etree.SubElement(sf, qn(A, "srgbClr"), val=fill)
    else:
        etree.SubElement(sppr, qn(A, "noFill"))
    if line:
        ln = etree.SubElement(sppr, qn(A, "ln"), w="10000")
        sf = etree.SubElement(ln, qn(A, "solidFill"))
        etree.SubElement(sf, qn(A, "srgbClr"), val=line)
    else:
        etree.SubElement(sppr, qn(A, "ln")).append(etree.Element(qn(A, "noFill")))
    tx = etree.SubElement(sp, qn(P, "txBody"))
    etree.SubElement(tx, qn(A, "bodyPr"), wrap="square", lIns=emu(margin), rIns=emu(margin), tIns=emu(margin), bIns=emu(margin))
    etree.SubElement(tx, qn(A, "lstStyle"))
    for item in paragraphs:
        if isinstance(item, str):
            item = {"text": item}
        add_paragraph(tx, item["text"], size=item.get("size", font_size), bold=item.get("bold", False),
                      color=item.get("color", color), bullet=item.get("bullet", False),
                      space_after=item.get("space_after", 220))
    insert_before_ext(tree, sp)
    return sp


def add_rect(tree, sid, name, x, y, w, h, fill, line=None):
    return add_textbox(tree, sid, name, x, y, w, h, [], fill=fill, line=line, margin=0)


def replace_body_with_paragraphs(tree, shape_id, paragraphs):
    sp = clear_shape_text_by_id(tree, shape_id)
    tx = sp.find(qn(P, "txBody"))
    for item in paragraphs:
        add_paragraph(tx, item["text"], size=item.get("size", 15), bold=item.get("bold", False),
                      color=item.get("color", "222222"), bullet=item.get("bullet", False),
                      space_after=item.get("space_after", 300))


def delete_graphic_frames(tree):
    for gf in tree.xpath("//p:graphicFrame", namespaces=NS):
        gf.getparent().remove(gf)


def draw_bar_chart(tree, start_id, x, y, w, h, title, labels, values, color, ymin=0.45, ymax=0.90):
    sid = start_id
    add_textbox(tree, sid, "chart title", x, y, w, 0.28, [{"text": title, "bold": True, "size": 12}], margin=0.02)
    sid += 1
    plot_x, plot_y, plot_w, plot_h = x + 0.20, y + 0.48, w - 0.35, h - 1.05
    # baseline and guide lines
    for tick in [0.50, 0.60, 0.70, 0.80, 0.90]:
        ty = plot_y + plot_h * (1 - (tick - ymin) / (ymax - ymin))
        add_rect(tree, sid, "grid", plot_x, ty, plot_w, 0.006, "D8DEE8")
        sid += 1
        add_textbox(tree, sid, "tick", x, ty - 0.09, 0.35, 0.16, [{"text": f"{tick:.2f}", "size": 7, "color": "697386"}], margin=0)
        sid += 1
    n = len(labels)
    gap = 0.15
    bw = (plot_w - gap * (n + 1)) / n
    for i, (lab, val) in enumerate(zip(labels, values)):
        bx = plot_x + gap + i * (bw + gap)
        bh = max(0.02, plot_h * (val - ymin) / (ymax - ymin))
        by = plot_y + plot_h - bh
        add_rect(tree, sid, "bar", bx, by, bw, bh, color)
        sid += 1
        add_textbox(tree, sid, "value", bx - 0.03, by - 0.22, bw + 0.06, 0.18, [{"text": f"{val:.3f}", "size": 8, "bold": True, "color": "222222"}], margin=0)
        sid += 1
        add_textbox(tree, sid, "label", bx - 0.12, plot_y + plot_h + 0.08, bw + 0.24, 0.34, [{"text": lab, "size": 7, "color": "333333"}], margin=0)
        sid += 1
    return sid


def draw_table(tree, start_id, x, y, col_widths, row_h, rows, header_fill="1E3A5F"):
    sid = start_id
    for r, row in enumerate(rows):
        cx = x
        for c, text in enumerate(row):
            fill = header_fill if r == 0 else ("F5F8FC" if r % 2 else "FFFFFF")
            color = "FFFFFF" if r == 0 else "222222"
            add_textbox(tree, sid, "table cell", cx, y + r * row_h, col_widths[c], row_h,
                        [{"text": text, "size": 8 if r else 8, "bold": r == 0, "color": color}],
                        fill=fill, line="C9D2DF", margin=0.04)
            sid += 1
            cx += col_widths[c]
    return sid


def main():
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    UNPACKED.mkdir(parents=True)
    with ZipFile(SRC) as zf:
        zf.extractall(UNPACKED)

    # Slide 4: smoother goal and tasks.
    tree = parse(4)
    replace_body_with_paragraphs(tree, "187", [
        {"text": "Цель: разработать программную систему для анализа связи между новостными данными и краткосрочной динамикой цен акций российского фондового рынка.", "bold": True, "size": 15, "space_after": 420},
        {"text": "Задачи:", "bold": True, "size": 15, "space_after": 170},
        {"text": "проанализировать подходы к новостному анализу финансовых рынков;", "bullet": True, "size": 14},
        {"text": "сформировать датасет из новостей и исторических котировок MOEX;", "bullet": True, "size": 14},
        {"text": "выделить рыночные, новостные и текстовые признаки;", "bullet": True, "size": 14},
        {"text": "обучить и сравнить модели для задач «События 24ч» и «Доходность 24ч»;", "bullet": True, "size": 14},
        {"text": "реализовать интерфейс для визуализации новостей, котировок и сигналов моделей.", "bullet": True, "size": 14},
    ])
    save(tree, 4)

    # Slide 8: news features.
    tree = parse(8)
    start = max_shape_id(tree) + 1
    card_y = 1.18
    cards = [
        ("Интенсивность потока", ["количество новостей за час", "число источников", "привязка к тикеру и времени"], "EAF2FF", "2E75B6"),
        ("Тональность", ["positive / negative / neutral", "sentiment_score", "уверенность sentiment-модели"], "EEF7ED", "4C9A2A"),
        ("Содержание новости", ["TF-IDF + TruncatedSVD", "RuBERT-эмбеддинги", "темы, события, релевантность"], "FFF3E6", "D97706"),
    ]
    x = 0.55
    for title, bullets, fill, accent in cards:
        add_textbox(tree, start, "feature card", x, card_y, 2.85, 2.45,
                    [{"text": title, "bold": True, "size": 12, "color": accent, "space_after": 250}] +
                    [{"text": b, "bullet": True, "size": 10, "space_after": 170} for b in bullets],
                    fill=fill, line="D6DEE8", radius=True, margin=0.12)
        start += 1
        x += 3.05
    add_textbox(tree, start, "feature process", 0.75, 4.08, 8.65, 0.63,
                [{"text": "Агрегация выполняется по паре «тикер - час»: в модель попадают только признаки, доступные на момент прогноза.", "size": 13, "bold": True, "color": "1F2937"}],
                fill="F6F8FB", line="D6DEE8", radius=True, margin=0.10)
    save(tree, 8)

    # Slide 10: model comparison charts.
    tree = parse(10)
    delete_graphic_frames(tree)
    sid = max_shape_id(tree) + 1
    labels = ["LogReg", "ExtraTrees", "CatBoost", "HGBoost"]
    roc = [0.669, 0.662, 0.680, 0.669]
    p5 = [0.839, 0.830, 0.875, 0.866]
    sid = draw_bar_chart(tree, sid, 0.65, 1.05, 4.25, 3.65, "ROC-AUC: лучшее значение по модели", labels, roc, "3B82F6", ymin=0.45, ymax=0.90)
    sid = draw_bar_chart(tree, sid, 5.05, 1.05, 4.25, 3.65, "Precision@5%: качество самых уверенных сигналов", labels, p5, "10B981", ymin=0.45, ymax=0.90)
    add_textbox(tree, sid, "chart note", 0.80, 4.72, 8.25, 0.38,
                [{"text": "CatBoost показывает максимум по ROC-AUC и Precision@5%; HistGradientBoosting близок по качеству и дает лучший F1 на объединенных признаках.", "size": 10, "color": "384152"}],
                fill="F6F8FB", line="D6DEE8", radius=True, margin=0.07)
    save(tree, 10)

    # Slide 11: best algorithm by feature space.
    tree = parse(11)
    delete_graphic_frames(tree)
    replace_body_with_paragraphs(tree, "252", [{"text": "Лучший алгоритм по признаковым пространствам", "size": 24}])
    sid = max_shape_id(tree) + 1
    rows = [
        ["Признаковое пространство", "Лучший алгоритм", "F1", "ROC-AUC", "PR-AUC", "Precision@5%"],
        ["Только новости", "CatBoost", "0.483", "0.497", "0.511", "0.607"],
        ["Рынок в момент новости", "CatBoost", "0.541", "0.680", "0.688", "0.875"],
        ["Новости + рынок", "HistGradientBoosting", "0.551", "0.669", "0.681", "0.866"],
    ]
    sid = draw_table(tree, sid, 0.62, 1.32, [2.20, 2.05, 0.82, 1.05, 1.05, 1.30], 0.48, rows)
    add_textbox(tree, sid, "criterion note", 0.72, 3.55, 8.55, 0.96,
                [{"text": "Критерий выбора: Precision@5% как метрика качества наиболее уверенных сигналов; при равенстве учитывался F1.", "size": 12, "bold": True, "color": "1F2937", "space_after": 180},
                 {"text": "Новостные признаки сами по себе слабые, но в моменты появления новостей рыночные признаки дают устойчивый сигнал; объединение признаков повышает F1.", "size": 11, "color": "384152"}],
                fill="F6F8FB", line="D6DEE8", radius=True, margin=0.10)
    save(tree, 11)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
        for path in UNPACKED.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(UNPACKED).as_posix())
    print(OUT)


if __name__ == "__main__":
    main()
