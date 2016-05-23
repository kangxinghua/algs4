package edu.princeton.cs.algs4;

/******************************************************************************
 *  Compilation:  javac GraphX.java
 *  Execution:    java GraphX input.txt
 *  Dependencies: BagOfInts.java In.java StdOut.java
 *  Data files:   http://algs4.cs.princeton.edu/41graph/tinyG.txt
 *
 *  A graph, implemented using an array of bags. This verison uses
 *  an resizing array of primitive ints for memory efficiency.
 *  Parallel edges and self-loops allowed.
 *
 *  % java GraphX tinyG.txt
 *  13 vertices, 13 edges
 *  0: 6 2 1 5
 *  1: 0
 *  2: 0
 *  3: 5 4
 *  4: 5 6 3
 *  5: 3 4 0
 *  6: 0 4
 *  7: 8
 *  8: 7
 *  9: 11 10 12
 *  10: 9
 *  11: 9 12
 *  12: 11 9
 *
 *  % java GraphX mediumG.txt
 *  250 vertices, 1273 edges
 *  0: 225 222 211 209 204 202 191 176 163 160 149 114 97 80 68 59 58 49 44 24 15
 *  1: 220 203 200 194 189 164 150 130 107 72
 *  2: 141 110 108 86 79 51 42 18 14
 *  ...
 *
 ******************************************************************************/


/**
 *  The <tt>GraphX</tt> class represents an undirected graph of vertices
 *  named 0 through V-1.
 *  It supports the following operations: add an edge to the graph,
 *  iterate over all of the neighbors adjacent to a vertex.
 *  Parallel edges and self-loops are permitted.
 *  <p>
 *  For additional documentation, see <a href="http://algs4.cs.princeton.edu/41graph">Section 4.1</a> of
 *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 *
 *  @author Robert Sedgewick
 *  @author Kevin Wayne
 */
public class GraphX {
    private final int V;
    private int E;
    private BagOfInts[] adj;

    /**
     * Initializes an empty graph with V vertices and 0 edges.
     * @throws java.lang.IllegalArgumentException if V < 0
     */
    public GraphX(int V) {
        if (V < 0) throw new IllegalArgumentException("Number of vertices must be nonnegative");
        this.V = V;
        this.E = 0;
        adj = new BagOfInts[V];
        for (int v = 0; v < V; v++) {
            adj[v] = new BagOfInts();
        }
    }

    /**
     * Initializes a random graph with V vertices and E edges.
     * Expected running time is proportional to V + E.
     * @param V number of vertices
     * @param E number of edges
     * @throws java.lang.IllegalArgumentException if either V < 0 or E < 0
     */
    public GraphX(int V, int E) {
        this(V);
        if (E < 0) throw new IllegalArgumentException("Number of edges must be nonnegative");
        for (int i = 0; i < E; i++) {
            int v = StdRandom.uniform(V);
            int w = StdRandom.uniform(V);
            addEdge(v, w);
        }
    }

    /**
     * Initializes a graph from input stream.
     * The format is the number of vertices V, followed by the number of edges E,
     * followed by E pairs of vertices, with each entry separated by whitespace.
     * @param in the input stream
     */
    public GraphX(In in) {
        this(in.readInt());
        int E = in.readInt();
        for (int i = 0; i < E; i++) {
            int v = in.readInt();
            int w = in.readInt();
            addEdge(v, w);
        }
    }

    /**
     * Initializes a new graph that is a deep copy of G.
     * @param G the graph to copy
     */
    public GraphX(GraphX G) {
        this(G.V());
        this.E = G.E();
        for (int v = 0; v < G.V(); v++) {
            // reverse so that adjacency list is in same order as original
            Stack<Integer> reverse = new Stack<Integer>();
            for (int w : G.adj[v]) {
                reverse.push(w);
            }
            for (int w : reverse) {
                adj[v].add(w);
            }
        }
    }

    /**
     * Returns the number of vertices in this graph.
     * @return the number of vertices in this graph
     */
    public int V() {
        return V;
    }

    /**
     * Returns the number of edges in this graph.
     * @return the number of edges in this graph
     */
    public int E() {
        return E;
    }


    /**
     * Adds the undirected edge v-w to this graph.
     * @param v one vertex in the edge
     * @param w the other vertex in the edge
     * @throws java.lang.IndexOutOfBoundsException unless both 0 <= v < V and 0 <= w < V
     */
    public void addEdge(int v, int w) {
        if (v < 0 || v >= V) throw new IndexOutOfBoundsException();
        if (w < 0 || w >= V) throw new IndexOutOfBoundsException();
        E++;
        adj[v].add(w);
        adj[w].add(v);
    }


    /**
     * Returns the list of neighbors of vertex v as an Iterable.
     * @return the list of neighbors of vertex v as an Iterable
     * @throws java.lang.IndexOutOfBoundsException unless 0 <= v < V
     */
    public Iterable<Integer> adj(int v) {
        if (v < 0 || v >= V) throw new IndexOutOfBoundsException();
        return adj[v];
    }


    /**
     * Returns a string representation of the graph.
     * @return the number of vertices V, followed by the number of edges E,
     *   followed by the V adjacency lists
     */
    public String toString() {
        StringBuilder s = new StringBuilder();
        String NEWLINE = System.getProperty("line.separator");
        s.append(V + " vertices, " + E + " edges " + NEWLINE);
        for (int v = 0; v < V; v++) {
            s.append(v + ": ");
            for (int w : adj[v]) {
                s.append(w + " ");
            }
            s.append(NEWLINE);
        }
        return s.toString();
    }


    /**
     * Test client.
     */
    public static void main(String[] args) {
        In in = new In(args[0]);
        GraphX G = new GraphX(in);
        StdOut.println(G);
    }

}