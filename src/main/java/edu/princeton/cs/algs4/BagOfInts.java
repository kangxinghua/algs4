package edu.princeton.cs.algs4;

/******************************************************************************
 * Compilation:  javac BagOfInts.java
 * Execution:    java BagOfInts
 * Dependencies: StdIn.java StdOut.java
 * <p>
 * Bag implementation storing integers with a resizing array.
 ******************************************************************************/

import java.util.Iterator;
import java.util.NoSuchElementException;

/**
 *  The <tt>BagOfInts</tt> class represents a bag (or multiset) of
 *  integers. It supports insertion and iterating over the
 *  items in arbitrary order.
 *  <p>
 *  This implementation uses a resizing array.
 *  The <em>add</em> operation takes constant amortized time; the
 *  <em>isEmpty</em>, and <em>size</em> operations
 *  take constant time. Iteration takes time proportional to the number of items.
 *  <p>
 *  For additional documentation, see <a href="http://algs4.cs.princeton.edu/13stacks">Section 1.3</a> of
 *  <i>Algorithms, 4th Edition</i> by Robert Sedgewick and Kevin Wayne.
 *
 *  @author Robert Sedgewick
 *  @author Kevin Wayne
 */
public class BagOfInts implements Iterable<Integer> {
    private int[] a;      // array of items
    private int N;        // number of elements on stack

    /**
     * Initializes an empty bag.
     */
    public BagOfInts() {
        a = new int[2];
        N = 0;
    }

    /**
     * Is this bag empty?
     * @return true if this bag is empty; false otherwise
     */
    public boolean isEmpty() {
        return N == 0;
    }

    /**
     * Returns the number of items in this bag.
     * @return the number of items in this bag
     */
    public int size() {
        return N;
    }

    // resize the underlying array holding the elements
    private void resize(int capacity) {
        assert capacity >= N;
        int[] temp = new int[capacity];
        for (int i = 0; i < N; i++)
            temp[i] = a[i];
        a = temp;
    }

    /**
     * Adds the item to this bag.
     * @param item the item to add to this bag
     */
    public void add(int item) {
        if (N == a.length) resize(2 * a.length);    // double size of array if necessary
        a[N++] = item;                            // add item
    }


    /**
     * Returns an iterator that iterates over the items in the bag in arbitrary order.
     * @return an iterator that iterates over the items in the bag in arbitrary order
     */
    public Iterator<Integer> iterator() {
        return new ArrayIterator();
    }

    // an iterator, doesn't implement remove() since it's optional
    private class ArrayIterator implements Iterator<Integer> {
        private int i = 0;

        public boolean hasNext() {
            return i < N;
        }

        public void remove() {
            throw new UnsupportedOperationException();
        }

        public Integer next() {
            if (!hasNext()) throw new NoSuchElementException();
            return a[i++];
        }
    }

    /**
     * Unit tests the <tt>BagOfInts</tt> data type.
     */
    public static void main(String[] args) {
        BagOfInts bag = new BagOfInts();
        bag.add(3);
        bag.add(1);
        bag.add(4);
        bag.add(1);
        bag.add(5);

        for (int x : bag)
            StdOut.println(x);
    }

}
