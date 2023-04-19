(* Queue *)
type 'a queue = Q of 'a list * 'a list;;

let norm = function
  | Q ([], tls) -> Q (List.rev tls, [])
  | q -> q;;

let qnull = function
  | Q ([], []) -> true
  | _ -> false;;

let enq (Q (hds, tls)) x = norm (Q (hds, x::tls));;

exception Empty;;

let deq = function
| Q (x::hds, tls) -> norm (Q (hds, tls))
| _ -> raise Empty;;

let qempty = Q ([], []);;

let qhd = function
| Q (x::_, _) -> x
| _ -> raise Empty;;

let popleft q = 
  let x = qhd q in let new_q = deq q 
  in x, new_q;;

(* Functional array *)

type 'a array = 
  | Lf
  | Br of 'a * 'a array * 'a array;;

exception Subscript;;

let rec update = function
  | Lf, k, w ->
      if k = 1 then
        Br (w, Lf, Lf)
      else
        raise Subscript 
  | Br (v, t1, t2), k, w ->
      if k = 1 then
        Br (w, t1, t2)
      else if k mod 2 = 0 then
        Br (v, update (t1, k / 2, w), t2)
      else
        Br (v, t1, update (t2, k / 2, w));;

let rec sub = function 
| Lf, _ -> raise Subscript
| Br (v, t1, t2), 1 -> v
| Br (v, t1, t2), k when k mod 2 = 0 -> sub (t1, k / 2)
| Br (v, t1, t2), k -> sub (t2, k / 2);;

let get_value = function 
| Lf -> None
| Br(v, t1, t2) -> v;;

(* Graph utilities *)

(* from (vertex from 0), to, capacity, flow, id from 1, reverse_id*)
type edges = 
  | Edge of  int * int * int * int * int * int;;

let get_id (Edge(_, _, _, _, id, _)) = id;;
let from_vertex (Edge(_from, _, _, _, _, _)) = _from;;
let to_vertex (Edge(_, _to, _, _, _, _)) = _to;;
let add_flow (Edge(a, b, c, _flow, d, e)) x = Edge(a, b, c, _flow+x, d, e);;
let reverse_id id = if id mod 2 == 0 then id-1 else id+1;;
let remaining_capacity (Edge(_,_, _cap, _flow, _, _)) = _cap - _flow;;

let tmp = [
  (0,1,40);
  (0,3,20);
  (1,3,20);
  (1,2,30);
  (2,3,10)
];;

let tmp2 = [(0,3,17085);
(3,27,189225);
(27,15,63288);
(3,1,759330);
(15,61,414178);
(1,35,329440);
(3,94,583020);
(0,68,197298);
(35,79,826454);
(79,26,113169);
(79,67,46128);
(35,95,314100);
(95,2,144330);
(61,84,606276);
(84,31,533664);
(27,23,228683);
(2,22,41938);
(35,40,235512);
(1,80,186348);
(79,76,673456);
(15,93,53760);
(27,46,116987);
(79,55,23358);
(2,14,226238);
(61,62,641900);
(40,53,67104);
(76,91,18080);
(79,36,18210);
(76,29,63600);
(1,92,528);
(26,6,30618);
(68,65,164679);
(80,59,204435);
(22,58,113924);
(84,60,667234);
(15,11,314560);
(93,72,60540);
(91,42,64864);
(94,41,78672);
(55,38,7450);
(53,30,383474);
(95,7,378144);
(59,20,132550);
(76,44,501228);
(31,33,18296);
(38,82,699100);
(76,73,428168);
(30,77,577348);
(76,43,360308);
(84,71,219516);
(79,24,242208);
(67,37,397244);
(43,83,338688);
(95,75,94952);
(26,63,450300);
(31,50,121770);
(94,78,66774);
(63,81,425295);
(83,89,43204);
(80,64,154175);
(73,39,21280);
(71,25,36400);
(75,17,31994);
(72,45,138437);
(83,97,10650);
(91,69,210120);
(53,10,142410);
(27,9,109740);
(65,13,336050);
(92,34,283272);
(65,47,143118);
(47,57,670842);
(58,74,57928);
(2,88,23560);
(27,52,150960);
(35,70,412850);
(31,96,267540);
(75,32,142679);
(75,4,124960);
(88,19,801073);
(43,54,322368);
(92,21,167139);
(54,56,267267);
(32,16,16659);
(50,86,67580);
(80,12,345840);
(39,8,368874);
(52,49,546845);
(45,51,290168);
(60,85,359904);
(91,28,158116);
(45,48,3084);
(75,18,250523);
(6,5,339957);
(76,87,255360);
(79,66,617056);
(80,90,198496);
(9,98,699152);
(61,99,50268);
(67,32,88892);
(82,69,523625);
(38,27,358620);
(48,79,247150);
(73,3,618017);
(96,41,489);
(15,82,183690);
(18,35,162648);
(94,28,8520);
(50,93,667554);
(72,62,19116);
(21,48,25715);
(88,36,40712);
(8,41,23484);
(37,78,1664);
(19,92,519360);
(38,91,478080);
(67,35,28536);
(32,20,468534);
(78,14,825968);
(85,79,193629);
(96,42,158484);
(9,42,449445);
(8,24,132940);
(78,97,2225);
(76,38,494856);
(89,56,7872);
(91,85,57204);
(0,40,32648);
(31,94,48815);
(3,12,14525);
(32,98,151704);
(5,88,170148);
(73,63,109214);
(20,92,97524);
(81,88,117819);
(67,94,218337);
(35,66,511648);
(94,92,284171);
(88,97,17072);
(6,71,206205);
(9,83,67746);
(1,14,418095);
(24,65,510300);
(30,81,75900);
(83,39,505300);
(65,50,135366);
(30,22,901048);
(73,19,41076);
(84,90,444726);
(12,73,309582);
(47,72,198288);
(32,7,476470);
(6,12,447492);
(95,21,644787);
(60,86,413376);
(4,72,154440);
(55,89,46995);
(39,87,279105);
(31,82,233232);
(22,43,255424);
(84,99,490842);
(72,6,57022);
(38,20,334356);
(95,33,56800);
(31,3,733125);
(69,40,759782);
(12,84,36925);
(48,27,120888);
(53,31,256200);
(84,70,599874);
(74,71,745752);
(52,72,432174);
(72,69,257890);
(91,80,444988);
(75,80,22609);
(66,61,789351);
(3,51,32844);
(23,45,195975);
(33,99,83454);
(76,24,260160);
(1,21,341784);
(64,91,499310);
(32,14,580903);
(72,22,107787);
(82,29,280178);
(31,83,382230);
(78,92,160881);
(67,86,630240);
(92,44,215514);
(71,52,597958);
(5,19,88416);
(58,19,620682);
(62,84,166897);
(39,15,102355);
(69,41,481932);
(9,89,91377);
(43,13,51465);
(47,76,10168)]

(* Build required data structures *)

let build_edges_list _tmp = 
  let rec buildEdges_list_acc acc id = function 
    | [] -> acc
    | (_from, _to, _capacity)::xs -> 
      buildEdges_list_acc
        ((Edge(_to, _from, 0, 0, id+1, id))::(Edge(_from, _to, _capacity, 0, id, id+1))::acc) 
      (id+2) xs in
  buildEdges_list_acc [] 1 _tmp;;

let build_edges_array _edges_list = 
    
  let rec build_edges_array_acc (array: edges array) (_edges_list: edges list) (i: int) = match _edges_list with 
    | [] -> array
    | x::xs -> let new_array = update (array, i, x) in 
      build_edges_array_acc new_array xs (i+1) in 
  
  build_edges_array_acc Lf (List.rev _edges_list) 1;;

(* Assume vertices are all labelled from 0-n, with no space in between 0-n *)
let rec find_max_node (_edges_list, max) = match _edges_list with
  | [] -> max
  | Edge(x, y, _, _, _, _)::xs -> 
      let z = if x > y then x else y in let new_max = if z > max then z else max in 
      find_max_node (xs, new_max);;

let init_out_edges _edges_list = 
  let max = find_max_node (_edges_list, 0) + 2 in 
  let rec build_array array i = 
    if i = max then array else
      let new_array = update (array, i, []) in 
      build_array new_array (i+1) in 
  build_array Lf 1;;

(* edges leaving out of vertex *)
let build_out_edges _edges_list = 
  let rec build_out_array_acc (array: int list array) (_edges_list: edges list) = match _edges_list with 
    | [] -> array
    | x::xs -> 
      let i = from_vertex x in
      let old_list = sub (array, i+1) in
      let new_list = (get_id x)::old_list in 

      let new_array = update (array, i+1, new_list) in 
      build_out_array_acc new_array xs in 
  build_out_array_acc (init_out_edges _edges_list) _edges_list;;

let buildVisited _edges_list source = 
  let max = find_max_node (_edges_list, 0) + 2 in 
  let rec build_array array i = 
    if i = max then array else
      let new_array = update (array, i, false) in 
      build_array new_array (i+1) in 
  update (build_array Lf 1, source, true);;

(* Find_path *)

let add_edges_to_queue _edges_array __vertex_array _out_edges _queue vertex path sink = 
  let _edges = (sub (_out_edges, vertex+1)) in 
  let rec _add_to_queue _queue _vertex_array _ids = match _ids with 
  | [] -> _queue
  | _id::xs -> let _edge = sub (_edges_array, _id) in
      let _to = to_vertex _edge in 
      let new_queue = if (not (sub (_vertex_array, _to+1))) && (remaining_capacity _edge)>0 then enq _queue (_to, _id::path) else _queue in 
    _add_to_queue new_queue _vertex_array xs in 
  _add_to_queue _queue __vertex_array _edges;;

exception NoMorePaths;;

let bfs_path_wrapper _edges_array (__vertex_array: bool array) _out_edges queue sink = 
  let rec bfs_path _vertex_array _queue = 
    if qnull _queue then raise NoMorePaths
    (* path will store list of edges to sink *)
    else let ((current, path), new_queue) = popleft _queue in
    if current = sink then path
    else let _new_queue = add_edges_to_queue _edges_array _vertex_array _out_edges new_queue current path sink in 
      bfs_path (update (_vertex_array, current+1, true)) _new_queue in 
  bfs_path __vertex_array queue;;

let find_augmenting_path (source, sink) (_edges_array, _visited, _out_edges) = 
  List.rev (bfs_path_wrapper _edges_array _visited _out_edges (enq (Q([], [])) (source, [])) sink);;

let rec find_bottleneck _edges_array min_cap = function
  | [] -> min_cap
  | id::xs -> find_bottleneck _edges_array (min (remaining_capacity (sub (_edges_array, id))) min_cap) xs;;

let rec update_path _edges_array _bottleneck_flow = function
  | [] -> _edges_array
  | id::xs -> 
    let r = reverse_id id in 
    update_path (
      update (
        (update (_edges_array, id, add_flow (sub(_edges_array, id)) _bottleneck_flow)),
        r, add_flow (sub(_edges_array, r)) (-_bottleneck_flow))
    )
    _bottleneck_flow xs;;

let max_flow capacities source sink = 
  let edges_list = build_edges_list capacities in 
  let edges_array = build_edges_array edges_list in 
  let visited = buildVisited edges_list (source+1) in 
  let out_edges = build_out_edges edges_list in 
  let rec edmond_karps _edges_array (total:int) = 
    try 
      let augmenting_path = find_augmenting_path (source, sink) (_edges_array, visited, out_edges) in 
      let bottleneck_flow = find_bottleneck _edges_array Int.max_int augmenting_path in 
      edmond_karps (
        update_path _edges_array bottleneck_flow augmenting_path
      )
      (total + bottleneck_flow)
    with NoMorePaths -> total in 
  edmond_karps edges_array 0;;

let benchmark f = 
  let t = Sys.time() in 
  let rec timer f iters = 
    if iters = 0 then Sys.time() -. t 
    else timer f (iters-1) in 
  timer f 1;;
  

max_flow tmp 0 3;;
max_flow tmp2 0 99;;

benchmark (fun () -> max_flow tmp2 0 99);;
