/**
 * Javascript file for Simple Photoshop Demo
 */
var ierUrl = location.origin + "/ier";

var build_slot_dict = function (slot, value = null, conf = null) {
    var slot_dict = {};
    slot_dict['slot'] = slot;
    if (value != null) slot_dict['value'] = value;
    if (conf != null) slot_dict['conf'] = conf;
    return slot_dict;
}

var find_slot_with_key = function (key, slots) {
    for (var i = 0; i < slots.length; ++i) {
        var slot = slots[i];
        if (slot['slot'] == key) {
            return slot;
        }
    }
    return null;
}

var previewFile = function () {
    var preview = document.querySelector('img');
    var file = document.querySelector('input[type=file]').files[0];
    var reader = new FileReader();

    reader.addEventListener("load", function () {
        preview.src = reader.result;
        var b64_img_str = preview.src.split(',')[1];

        var observation = {
            'user_acts': [
                {
                    'dialogue_act': build_slot_dict('dialogue_act', 'inform', 1.0),
                    'intent': build_slot_dict('intent', 'load', 1.0),
                    'slots': [
                        build_slot_dict('b64_img_str', b64_img_str, 1.0)
                    ]
                }
            ]
        }
        submitRequest(observation);
    }, false);

    if (file) {
        reader.readAsDataURL(file);
    }
}

var getObservation = function () {
    // Parse slot values here
    var dialogue_act = $("#dialogueact").val();
    var intent = $('#intent').val();

    var slots = [];
    for (var i = 1; i <= 3; ++i) {
        var slot_id = "#slot" + i.toString();
        var value_id = "#value" + i.toString();
        var conf_id = "#conf" + i.toString();
        if ($(slot_id).val().trim() != "" && $(value_id).val().trim() != "" && $(conf_id).val().trim() != "") {
            var slot = $(slot_id).val();
            var value = $(value_id).val();
            var conf = parseFloat($(conf_id).val());
            var slot_dict = build_slot_dict(slot, value, conf);
            slots.push(slot_dict);
        }
    }

    var observation = {
        'user_acts': [
            {
                'dialogue_act': build_slot_dict('dialogue_act', dialogue_act, 1.0),
                'intent': build_slot_dict('intent', intent, 1.0),
                'slots': slots
            }
        ]
    };
    console.log("getObservation", observation);
    return observation;
}

var submitRequest = function (observation) {

    var data = {
        'observation': JSON.stringify(observation)
    };

    console.log('submit', data);

    toggleLoading(true);
    $.post(ierUrl, data, function (response) {
        console.log(response);
        // Load system utterance
        loadUtterance(response['system_utterance']);

        // There should only be 1 act
        var ps_act = response['photoshop_acts'][0];
        var ps_slots = ps_act['slots']

        var b64_img_str_slot = find_slot_with_key('b64_img_str', ps_slots);
        var masked_b64_img_str_slot = find_slot_with_key('masked_b64_img_str', ps_slots);

        var obj = {}
        if (b64_img_str_slot != null) obj['b64_img_str'] = b64_img_str_slot['value'];
        if (masked_b64_img_str_slot != null) obj['masked_b64_img_str'] = masked_b64_img_str_slot['value'];

        loadImage(obj);
    }).always(function () {
        toggleLoading(false);
    })
}

var toggleLoading = function (show) {
    if (show) {
        $("#loading-container").show();
    } else {
        $("#loading-container").hide();
    }
}

var loadImage = function (obj) {
    var image_names = ["masked_b64_img_str", "b64_img_str"];
    var image_ids = ["masked-image", "image"]
    for (var i = 0; i < image_names.length; ++i) {
        var image_name = image_names[i];
        var image_id = image_ids[i];
        if (image_name in obj) {
            var img_str = obj[image_name];
            $("#" + image_id).attr('src', "data:image/png;base64," + img_str);
        }
    }
}

var loadUtterance = function (utterance) {
    $("#system-message").text(utterance);
}


$(document).ready(function () {

    $("#submit").click(function () {
        observation = getObservation();
        submitRequest(observation);
    });
});