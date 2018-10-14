/*
 * decaffeinate suggestions:
 * DS101: Remove unnecessary use of Array.from
 * DS102: Remove unnecessary code created because of implicit returns
 * DS207: Consider shorter variations of null checks
 * Full docs: https://github.com/decaffeinate/decaffeinate/blob/master/docs/suggestions.md
 */
// Use coffee-script compiler to obtain a javascript file.
//
//    coffee -c bbox_annotator.coffee
//
// See http://coffeescript.org/

// BBox selection window.
class BBoxSelector {
    // Initializes selector in the image frame.
    constructor(image_frame, options) {
        if (options == null) {
            options = {};
        }
        if (!options.input_method) {
            options.input_method = "text";
        }
        this.image_frame = image_frame;
        this.border_width = options.border_width || 2;
        this.selector = $('<div class="bbox_selector"></div>');
        this.selector.css({
            "border": (this.border_width) + "px dotted rgb(127,255,127)",
            "position": "absolute"
        });
        this.image_frame.append(this.selector);
        this.selector.css({
            "border-width": this.border_width
        });
        this.selector.hide();
        this.create_label_box(options);
    }

    // Initializes a label input box.
    create_label_box(options) {
        if (!options.labels) {
            options.labels = ["object"];
        }
        this.label_box = $('<div class="label_box"></div>');
        this.label_box.css({
            "position": "absolute"
        });
        this.image_frame.append(this.label_box);
        switch (options.input_method) {
            case 'select':
                if (typeof options.labels === "string") {
                    options.labels = [options.labels];
                }
                this.label_input = $('<select class="label_input" name="label"></select>');
                this.label_box.append(this.label_input);
                this.label_input.append($('<option value>choose an item</option>'));
                for (let label of Array.from(options.labels)) {
                    this.label_input.append(`<option value="${label}">` +
                        label + '</option>'
                    );
                }
                this.label_input.change(function (e) {
                    return this.blur();
                });
                break;
            case 'text':
                if (typeof options.labels === "string") {
                    options.labels = [options.labels];
                }
                this.label_input = $('<input class="label_input" name="label" ' +
                    'type="text" value>');
                this.label_box.append(this.label_input);
                this.label_input.autocomplete({
                    source: options.labels || [''],
                    autoFocus: true
                });
                break;
            case 'fixed':
                if ($.isArray(options.labels)) {
                    options.labels = options.labels[0];
                }
                this.label_input = $('<input class="label_input" name="label" type="text">');
                this.label_box.append(this.label_input);
                this.label_input.val(options.labels);
                break;
            default:
                throw `Invalid label_input parameter: ${options.input_method}`;
        }
        return this.label_box.hide();
    }

    // Crop x and y to the image size.
    crop(pageX, pageY) {
        let point;
        return point = {
            x: Math.min(Math.max(Math.round(pageX - this.image_frame.offset().left), 0),
                Math.round(this.image_frame.width() - 1)),
            y: Math.min(Math.max(Math.round(pageY - this.image_frame.offset().top), 0),
                Math.round(this.image_frame.height() - 1))
        };
    }

    // When a new selection is made.
    start(pageX, pageY) {
        this.pointer = this.crop(pageX, pageY);
        this.offset = this.pointer;
        this.refresh();
        this.selector.show();
        $('body').css('cursor', 'crosshair');
        return document.onselectstart = () => false;
    }

    // When a selection updates.
    update_rectangle(pageX, pageY) {
        this.pointer = this.crop(pageX, pageY);
        return this.refresh();
    }

    // When starting to input label.
    input_label(options) {
        $('body').css('cursor', 'default');
        document.onselectstart = () => true;
        this.label_box.show();
        return this.label_input.focus();
    }

    // Finish and return the annotation.
    finish(options) {
        this.label_box.hide();
        this.selector.hide();
        const data = this.rectangle();
        data.label = $.trim(this.label_input.val().toLowerCase());
        if (options.input_method !== 'fixed') {
            this.label_input.val('');
        }
        return data;
    }

    // Get a rectangle.
    rectangle() {
        let rect;
        const x1 = Math.min(this.offset.x, this.pointer.x);
        const y1 = Math.min(this.offset.y, this.pointer.y);
        const x2 = Math.max(this.offset.x, this.pointer.x);
        const y2 = Math.max(this.offset.y, this.pointer.y);
        return rect = {
            left: x1,
            top: y1,
            width: (x2 - x1) + 1,
            height: (y2 - y1) + 1
        };
    }

    // Update css of the box.
    refresh() {
        const rect = this.rectangle();
        this.selector.css({
            left: (rect.left - this.border_width) + 'px',
            top: (rect.top - this.border_width) + 'px',
            width: rect.width + 'px',
            height: rect.height + 'px'
        });
        return this.label_box.css({
            left: (rect.left - this.border_width) + 'px',
            top: (rect.top + rect.height + this.border_width) + 'px'
        });
    }

    // Return input element.
    get_input_element() {
        return this.label_input;
    }
}

// Annotator object definition.
this.BBoxAnnotator = class BBoxAnnotator {
    // Initialize the annotator layout and events.
    constructor(options) {
        const annotator = this;
        this.annotator_element = $(options.id || "#bbox_annotator");
        this.border_width = options.border_width || 2;
        this.show_label = options.show_label || (options.input_method !== "fixed");

        if (options.multiple != null) {
            this.multiple = options.multiple;
        } else {
            this.multiple = true;
        }

        this.image_frame = $('<div class="image_frame"></div>');
        this.annotator_element.append(this.image_frame);
        const image_element = new Image();
        image_element.src = options.url;
        image_element.onload = function () {
            if (!options.width) {
                options.width = image_element.width;
            }
            if (!options.height) {
                options.height = image_element.height;
            }
            annotator.annotator_element.css({
                "width": (options.width + (annotator.border_width * 2)) + 'px',
                "height": (options.height + (annotator.border_width * 2)) + 'px',
                "cursor": "crosshair"
            });
            annotator.image_frame.css({
                "background-image": `url('${image_element.src}')`,
                "width": options.width + "px",
                "height": options.height + "px",
                "position": "relative"
            });
            annotator.selector = new BBoxSelector(annotator.image_frame, options);
            return annotator.initialize_events(annotator.selector, options);
        };
        image_element.onerror = () => annotator.annotator_element.text(`Invalid image URL: ${options.url}`);
        this.entries = [];
        this.onchange = options.onchange;
    }

    // Initialize events.
    initialize_events(selector, options) {
        let status = 'free';
        this.hit_menuitem = false;
        const annotator = this;
        this.annotator_element.mousedown(function (e) {
            if (!annotator.hit_menuitem) {
                switch (status) {
                    case 'free':
                    case 'input':
                        if (status === 'input') {
                            selector.get_input_element().blur();
                        }
                        if (e.which === 1) { // left button
                            selector.start(e.pageX, e.pageY);
                            status = 'hold';
                        }
                        break;
                }
            }
            annotator.hit_menuitem = false;
            return true;
        });
        $(window).mousemove(function (e) {
            switch (status) {
                case 'hold':
                    selector.update_rectangle(e.pageX, e.pageY);
                    break;
            }
            return true;
        });
        $(window).mouseup(function (e) {
            switch (status) {
                case 'hold':
                    selector.update_rectangle(e.pageX, e.pageY);
                    selector.input_label(options);
                    status = 'input';
                    if (options.input_method === 'fixed') {
                        selector.get_input_element().blur();
                    }
                    break;
            }
            return true;
        });
        selector.get_input_element().blur(function (e) {
            switch (status) {
                case 'input':
                    var data = selector.finish(options);
                    if (data.label) {
                        annotator.add_entry(data);
                        if (annotator.onchange) {
                            annotator.onchange(annotator.entries);
                        }
                    }
                    status = 'free';
                    break;
            }
            return true;
        });
        selector.get_input_element().keypress(function (e) {
            switch (status) {
                case 'input':
                    if (e.which === 13) {
                        selector.get_input_element().blur();
                    }
                    break;
            }
            return e.which !== 13;
        });
        selector.get_input_element().mousedown(e => annotator.hit_menuitem = true);
        selector.get_input_element().mousemove(e => annotator.hit_menuitem = true);
        selector.get_input_element().mouseup(e => annotator.hit_menuitem = true);
        return selector.get_input_element().parent().mousedown(e => annotator.hit_menuitem = true);
    }

    // Add a new entry.
    add_entry(entry) {
        if (!this.multiple) {
            this.annotator_element.find(".annotated_bounding_box").detach();
            this.entries.splice(0);
        }

        this.entries.push(entry);
        const box_element = $('<div class="annotated_bounding_box"></div>');
        box_element.appendTo(this.image_frame).css({
            "border": this.border_width + "px solid rgb(127,255,127)",
            "position": "absolute",
            "top": (entry.top - this.border_width) + "px",
            "left": (entry.left - this.border_width) + "px",
            "width": entry.width + "px",
            "height": entry.height + "px",
            "color": "rgb(127,255,127)",
            "font-family": "monospace",
            "font-size": "small"
        });
        const close_button = $('<div></div>').appendTo(box_element).css({
            "position": "absolute",
            "top": "-8px",
            "right": "-8px",
            "width": "16px",
            "height": "0",
            "padding": "16px 0 0 0",
            "overflow": "hidden",
            "color": "#fff",
            "background-color": "#030",
            "border": "2px solid #fff",
            "-moz-border-radius": "18px",
            "-webkit-border-radius": "18px",
            "border-radius": "18px",
            "cursor": "pointer",
            "-moz-user-select": "none",
            "-webkit-user-select": "none",
            "user-select": "none",
            "text-align": "center"
        });
        $("<div></div>").appendTo(close_button).html('&#215;').css({
            "display": "block",
            "text-align": "center",
            "width": "16px",
            "position": "absolute",
            "top": "-2px",
            "left": "0",
            "font-size": "16px",
            "line-height": "16px",
            "font-family": '"Helvetica Neue", Consolas, Verdana, Tahoma, Calibri, ' +
                'Helvetica, Menlo, "Droid Sans", sans-serif'
        });
        const text_box = $('<div></div>').appendTo(box_element).css({
            "overflow": "hidden"
        });
        if (this.show_label) {
            text_box.text(entry.label);
        }
        const annotator = this;
        box_element.hover((e => close_button.show()), (e => close_button.hide()));
        close_button.mousedown(e => annotator.hit_menuitem = true);
        close_button.click(function (e) {
            const clicked_box = close_button.parent(".annotated_bounding_box");
            const index = clicked_box.prevAll(".annotated_bounding_box").length;
            clicked_box.detach();
            annotator.entries.splice(index, 1);
            return annotator.onchange(annotator.entries);
        });
        return close_button.hide();
    }
    // Clear all entries.
    clear_all(e) {
        this.annotator_element.find(".annotated_bounding_box").detach();
        this.entries.splice(0);
        return this.onchange(this.entries);
    }
};