cd /opt/subs-gp/clip-vector-api/ \
&& curl -s -X POST 'http://localhost:8200/v1/vector' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -H 'Authorization: Basic dXNlcjpwYXNz' \
  -F 'file=@3-pieces.jpg' \
| awk '{json = json $0 "\n"} END {
    if (match(json, /"status":[ \t]*200/)) {
        new_json = "";
        while (match(json, /"(piece|vector)":[ \t]*\[[^\]]*\]/)) {
            match_start = RSTART;
            match_len = RLENGTH;
            before = substr(json, 1, match_start - 1);
            full_match = substr(json, match_start, match_len);
            match(full_match, /"(piece|vector)"/);
            key_name = substr(full_match, RSTART, RLENGTH);
            idx = index(full_match, "[");
            inner = substr(full_match, idx + 1, length(full_match) - idx - 1);
            n = split(inner, a, /,/);
            out = key_name ": [";
            limit = (n > 10 ? 10 : n);
            for(i=1; i<=limit; i++) {
                gsub(/^[ \t\n\r]+|[ \t\n\r]+$/, "", a[i]);
                out = out a[i] (i < limit ? ", " : "");
            }
            out = out "]";
            new_json = new_json before out;
            json = substr(json, match_start + match_len);
        }
        json = new_json json;
    }
    gsub(/\n$/, "", json);
    print json;
}'