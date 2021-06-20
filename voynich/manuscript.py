import re
import collections


Page = collections.namedtuple(
    "Page", ["page_id", "folio_num", "folio_side", "folio_qualifier", "bodies"]
)
Body = collections.namedtuple(
    "Body", ["body_id", "page_id", "body_type", "body_qualifier", "lines"]
)
Line = collections.namedtuple(
    "Line", ["line_id", "body_id", "page_id", "line_number", "line_qualifier", "words"]
)


class VoynichManuscript:
    def __init__(self, transcription_txt_filepath):
        self.transcription_txt_filepath = transcription_txt_filepath
        self.pages = collections.OrderedDict()

        self.build()

    def build(self):
        # load transcription
        with open(self.transcription_txt_filepath, "r") as f:
            txt_lines = [line.strip() for line in f.readlines()]

        # process transcription
        for txt_line in txt_lines:

            # extract annotation from line
            txt_line = "".join(txt_line.split())
            annotation = re.search("(<.*>)", txt_line).groups()[0].strip()

            #
            new_folio_indicator_line_pattern = r"^<f([0-9]+)([rv])([0-9]*[a-z]*)?>$"
            new_folio_indicator_line_match = re.match(
                new_folio_indicator_line_pattern, annotation
            )

            #
            text_line_pattern = r"^<f([0-9]+)([rv])([0-9]*[a-z]*)?\.([A-Za-z])([0-9]*)?\.([0-9]+)([a-z]*)?;([A-Z])>"
            text_line_match = re.match(text_line_pattern, annotation)

            # get pages (Page --> Body --> Line)

            if new_folio_indicator_line_match:
                # get annotation components
                (
                    folio_num,
                    folio_side,
                    folio_qualifier,
                ) = new_folio_indicator_line_match.groups()

                # add page
                page_id = f"f{folio_num}{folio_side}{folio_qualifier}"
                page = Page(
                    page_id,
                    folio_num,
                    folio_side,
                    folio_qualifier,
                    collections.OrderedDict(),
                )
                self.pages[page_id] = page

            elif text_line_match:
                # get annotation components
                (
                    folio_num,
                    folio_side,
                    folio_qualifier,
                    body_type,
                    body_qualifier,
                    line_number,
                    line_qualifier,
                    transcriber_tag,
                ) = text_line_match.groups()

                # get words
                words = [
                    word
                    for word in txt_line.replace(annotation, "").strip().split(".")
                    if word
                ]

                # add body
                page_id = f"f{folio_num}{folio_side}{folio_qualifier}"
                body_id = f"{page_id}.{body_type}{body_qualifier}"
                if body_id not in self.pages[page_id].bodies:
                    self.pages[page_id].bodies[body_id] = Body(
                        body_id,
                        page_id,
                        body_type,
                        body_qualifier,
                        collections.OrderedDict(),
                    )

                # add line
                line_id = f"{body_id}.{line_number}{line_qualifier}"
                line = Line(
                    line_id, body_id, page_id, line_number, line_qualifier, words
                )
                self.pages[page_id].bodies[body_id].lines[line_id] = line

            else:
                raise Exception(
                    f"Annotation does not match any known transcription regex pattern: {annotation}"
                )
