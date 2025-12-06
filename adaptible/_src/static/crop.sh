#!/bin/bash

# If not installed, install imagemagick (brew install imagemagick)
_INPUT_FILE_PATH="";
_OUTPUT_FILE_PATH="";
_PERCENTAGE="";
_REMOVE_BG=0;

parse_flags() {
    while [[ "$#" -gt 0 ]]; do
        case "$1" in

            -i|--input) # Handle the -i flag
                _INPUT_FILE_PATH="${2}"
                echo "Input file path requested." >&2
                shift 2
                ;;
            -o|--output) # Handle the -o flag
                _OUTPUT_FILE_PATH="${2}"
                echo "Output file path requested." >&2
                shift 2
                ;;
            -p|--percentage) # Handle the -p flag with an argument
                _PERCENTAGE="${2}"
                echo "Percentage requested." >&2
                shift 2
                ;;
            -r|--remove-bg) # Handle the -r flag
                _REMOVE_BG=1;
                echo "Remove background flag set." >&2
                shift 1
                ;;
            *) # Handle invalid options
                exit 1
                ;;
        esac
    done
    
    # Shift the arguments so that positional parameters ($1, $2, etc.) 
    # after the flags are accessible.
    shift $((OPTIND-1)) 
}

function main() {
    parse_flags "$@"

    if [ -z "${_INPUT_FILE_PATH}" ] || [ -z "${_OUTPUT_FILE_PATH}" ] || [ -z "${_PERCENTAGE}" ]; then
        echo "Usage: crop.sh -i <_INPUT_FILE_PATH> -o <_OUTPUT_FILE_PATH> -p <percentage> [-r]" >&2
        exit 1
    fi
    local original_image_width="$(identify -format "%w" "$_INPUT_FILE_PATH")"
    local original_image_height="$(identify -format "%h" "$_INPUT_FILE_PATH")"
    local new_image_width="$((original_image_width * _PERCENTAGE / 100))"
    local new_image_height="$((original_image_height * _PERCENTAGE / 100))"
    local offset_x="$(( (original_image_width - new_image_width) / 2 ))"
    local offset_y="$(( (original_image_height - new_image_height) / 2 ))"
    convert "${_INPUT_FILE_PATH}" +repage -crop \
        "${new_image_width}x${new_image_height}+${offset_x}+${offset_y}" \
        +repage "${_OUTPUT_FILE_PATH}"
    if [ "${_REMOVE_BG}" -eq 1 ]; then
        remove_background "${_OUTPUT_FILE_PATH}"
    fi
}

function remove_background() {
    local output_image_path="$1"
    # Move to a temp file
    local input_image_path="${output_image_path%.*}_tmp.png"
    mv "${output_image_path}" "${input_image_path}"
    
    local args=(
        -fuzz 5%
        -fill none
    );
    # Find a handful of pixels in the corners to determine background color
    for coord in "0,0" "0,$(( $(identify -format "%h" "${input_image_path}") - 1 ))" \
        "$(( $(identify -format "%w" "${input_image_path}") - 1 )),0" \
        "$(( $(identify -format "%w" "${input_image_path}") - 1 )),$(( $(identify -format "%h" "${input_image_path}") - 1 ))"; do
        local color;
        color="$(magick "${input_image_path}" -format "%[pixel:p{${coord}}]" info:)"
        args+=("-opaque" "${color}");
    done
    echo "Args: " "${args[@]}"
    
    convert "${input_image_path}" "${args[@]}" "${output_image_path}"
    rm "${input_image_path}"
}

main "$@"