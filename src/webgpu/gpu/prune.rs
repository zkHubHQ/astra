use regex::Regex;
use std::collections::{HashMap, HashSet};

fn prune(shader_code: &str, function_names: &[&str]) -> String {
    let mut shader_code = Regex::new(r"//.*|/\*[\s\S]*?\*/")
        .unwrap()
        .replace_all(shader_code, "")
        .to_string();

    let function_pattern =
        Regex::new(r"(?s)fn\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\((.*?)\)\s*->\s*(.*?)\s*\{").unwrap();
    let global_structs_and_consts_pattern = Regex::new(r"(?s)(struct\s+[a-zA-Z_][a-zA-Z_0-9]*\s*\{.*?\})|(\bconst\b\s+[a-zA-Z_][a-zA-Z_0-9]*\s*:\s*.*?;)|(\balias\b\s+[a-zA-Z_][a-zA-Z_0-9]*\s*=\s*.*?;)").unwrap();

    let mut global_structs_and_consts = String::new();
    for cap in global_structs_and_consts_pattern.captures_iter(&shader_code) {
        global_structs_and_consts.push_str(&cap[0]);
        global_structs_and_consts.push_str("\n\n");
    }

    let mut functions = HashMap::new();
    for cap in function_pattern.captures_iter(&shader_code) {
        let func_name = &cap[1];
        let mut brace_count = 1;
        let start_index = cap.get(0).unwrap().end();
        let mut end_index = start_index;
        while brace_count > 0 && end_index < shader_code.len() {
            match &shader_code[end_index..=end_index] {
                "{" => brace_count += 1,
                "}" => brace_count -= 1,
                _ => {}
            }
            end_index += 1;
        }

        let func_body = &shader_code[cap.get(0).unwrap().start()..end_index];
        functions.insert(func_name.to_string(), func_body.to_string());
    }

    let mut used_functions = HashSet::new();
    for &name in function_names {
        used_functions.insert(name.to_string());
    }

    fn add_called_functions(
        function_body: &str,
        functions: &HashMap<String, String>,
        used_functions: &mut HashSet<String>,
    ) {
        for (name, body) in functions {
            if function_body.contains(name) && !used_functions.contains(name) {
                used_functions.insert(name.clone());
                add_called_functions(body, functions, used_functions); // Recursive call
            }
        }
    }

    for &fn_name in function_names {
        if let Some(body) = functions.get(fn_name) {
            add_called_functions(body, &functions, &mut used_functions);
        }
    }

    let mut pruned_code = global_structs_and_consts;
    for used_function in used_functions.iter() {
        if let Some(body) = functions.get(used_function) {
            pruned_code.push_str(body);
            pruned_code.push_str("\n\n");
        }
    }

    pruned_code.trim_end_matches('\n').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn extract_function_names(code: &str) -> HashSet<String> {
        let function_pattern = Regex::new(r"fn\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(").unwrap();
        function_pattern.captures_iter(code)
            .map(|cap| cap[1].to_string())
            .collect()
    }

    #[test]
    fn test_prune_shader_code() {
        let shader_code = r#"
        // A comment line
        fn unusedFunction() -> void {
            // This function is not used
        }

        fn usedFunction() -> void {
            // This function is used and should be retained
        }

        fn main() -> void {
            usedFunction();
        }

        fn anotherUnusedFunction() -> void {
            // This function is also unused
        }
        "#;

        let function_names = ["main"];
        let pruned_code = prune(shader_code, &function_names);

        let expected_functions = ["usedFunction", "main"].iter().cloned().map(String::from).collect::<HashSet<_>>();
        let actual_functions = extract_function_names(&pruned_code);

        assert_eq!(actual_functions, expected_functions);
    }
}


