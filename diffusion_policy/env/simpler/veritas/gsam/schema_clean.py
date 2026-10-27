# Vendored verbatim from veritas@8896225 src/agent/utils/gemini_client.py GeminiClient._clean_schema_for_gemini
# (dedented to a module function; self unused). Do not edit; re-vendor instead.

def clean_schema_for_gemini(schema_dict: dict) -> dict:
    """
    Clean JSON schema to remove $defs and inline references for Gemini compatibility.
    
    Args:
        schema_dict: Pydantic-generated JSON schema
    
    Returns:
        Cleaned schema dict compatible with Gemini
    """
    import copy
    
    # Fields that Gemini's Schema proto doesn't support
    # Keep: type, properties, items, required, enum (essential structure)
    UNSUPPORTED_FIELDS = {
        # JSON Schema structure
        "$defs", "$schema", "$ref", "$id", "$anchor",
        # Composition
        "anyOf", "oneOf", "allOf", "not",
        # Metadata
        "title", "description", "default", "examples", "const",
        # Numeric validation (from Field(ge=..., le=...))
        "maximum", "minimum", "exclusiveMaximum", "exclusiveMinimum",
        "multipleOf",
        # String validation
        "maxLength", "minLength", "pattern",
        # Array validation
        "maxItems", "minItems", "uniqueItems", "prefixItems",
        # Object validation
        "maxProperties", "minProperties", "additionalProperties", "additionalItems",
        # Format
        "format",
    }
    
    def _inline_refs(obj, defs_map):
        """Recursively inline $ref references and remove unsupported fields."""
        if isinstance(obj, dict):
            # Handle anyOf/oneOf/allOf - convert to simpler structure
            if "anyOf" in obj:
                # Take first option (usually the non-null one)
                any_of = obj["anyOf"]
                if any_of and len(any_of) > 0:
                    # Process first option and merge with other fields
                    first_option = _inline_refs(any_of[0], defs_map)
                    if isinstance(first_option, dict):
                        result = copy.deepcopy(first_option)
                        # Add other fields from obj (excluding anyOf and unsupported fields)
                        for key, value in obj.items():
                            if key not in UNSUPPORTED_FIELDS and key != "anyOf":
                                result[key] = _inline_refs(value, defs_map)
                        return result
                    return first_option
                # Empty anyOf, remove it
                obj = {k: v for k, v in obj.items() if k != "anyOf"}
            
            if "oneOf" in obj:
                # Take first option
                one_of = obj["oneOf"]
                if one_of and len(one_of) > 0:
                    first_option = _inline_refs(one_of[0], defs_map)
                    if isinstance(first_option, dict):
                        result = copy.deepcopy(first_option)
                        for key, value in obj.items():
                            if key not in UNSUPPORTED_FIELDS and key != "oneOf":
                                result[key] = _inline_refs(value, defs_map)
                        return result
                    return first_option
                obj = {k: v for k, v in obj.items() if k != "oneOf"}
            
            if "allOf" in obj:
                # Merge all options
                all_of = obj["allOf"]
                if all_of and len(all_of) > 0:
                    result = {}
                    for option in all_of:
                        processed_option = _inline_refs(option, defs_map)
                        if isinstance(processed_option, dict):
                            result.update(processed_option)
                    # Add other fields from obj
                    for key, value in obj.items():
                        if key not in UNSUPPORTED_FIELDS and key != "allOf":
                            result[key] = _inline_refs(value, defs_map)
                    return result
                obj = {k: v for k, v in obj.items() if k != "allOf"}
            
            # First, process all values recursively (this handles nested $ref)
            processed = {}
            for key, value in obj.items():
                if key in UNSUPPORTED_FIELDS:
                    # Skip unsupported fields (we'll handle $ref separately)
                    continue
                else:
                    processed[key] = _inline_refs(value, defs_map)
            
            # Now handle $ref if present
            if "$ref" in obj:
                ref_path = obj["$ref"]
                if ref_path.startswith("#/$defs/"):
                    ref_name = ref_path.split("/")[-1]
                    if ref_name in defs_map:
                        # Get the referenced definition and inline it
                        ref_def = _inline_refs(defs_map[ref_name], defs_map)
                        # Merge with other fields (processed fields take precedence over ref_def)
                        if isinstance(ref_def, dict):
                            # Merge: ref_def first, then processed (processed overrides)
                            merged = copy.deepcopy(ref_def)
                            merged.update(processed)
                            return merged
                        else:
                            # If ref_def is not a dict, use processed if available, else ref_def
                            return processed if processed else ref_def
                # If ref not found or not a $defs ref, just remove it and return processed
                return processed
            
            # No $ref, return processed dict
            return processed
        elif isinstance(obj, list):
            return [_inline_refs(item, defs_map) for item in obj]
        else:
            return obj
    
    # Extract $defs if present
    defs_map = schema_dict.get("$defs", {})
    
    # Create a copy without $defs
    cleaned = copy.deepcopy(schema_dict)
    cleaned.pop("$defs", None)
    
    # Inline all references
    cleaned = _inline_refs(cleaned, defs_map)
    
    return cleaned
