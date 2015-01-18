import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.WeightedPropertyVectorWritable;
import org.apache.mahout.clustering.kmeans.KMeansDriver;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.EuclideanDistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class App {

	public static final double[][] points = { { 1, 1 }, { 2, 1 }, { 1, 2 },
			{ 2, 2 }, { 3, 3 }, { 8, 8 }, { 9, 8 }, { 8, 9 }, { 9, 9 } };

	// Write data to sequence files in Hadoop (write the vector to sequence
	// file)
	private static HashMap<Integer, String> conv = new HashMap<Integer, String>();
	static {
		conv.put(0, "zero");
		conv.put(1, "one");
		conv.put(2, "two");
		conv.put(3, "three");
		conv.put(4, "four");
		conv.put(5, "five");
		conv.put(6, "six");
		conv.put(7, "seven");
		conv.put(8, "eight");
		conv.put(9, "nine");
		conv.put(10,"ten");
	}
	
	public static void writePointsToFile(List<NamedVector> points, String fileName,
			FileSystem fs, Configuration conf) throws IOException {

		Path path = new Path(fileName);
		SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
				Text.class, VectorWritable.class);
		long recNum = 0;
		VectorWritable vec = new VectorWritable();

		for (NamedVector point : points) {
			vec.set(point);
			recNum++;
			writer.append(new Text(point.getName()), vec);
		}

		writer.close();
	}

	// Read the points to vector from 2D array
	public static List getPoints(double[][] raw) {
		List points = new ArrayList();
		for (int i = 0; i < raw.length; i++) {
			double[] fr = raw[i];
			Vector vec = new RandomAccessSparseVector(fr.length);
			vec.assign(fr);
			System.out.println(conv.get(i));
			NamedVector nv = new NamedVector(vec,conv.get(new Integer(i)));
			points.add(nv);
		}
		return points;
	}

	public static void main(String args[]) throws Exception {

		// specify the number of clusters
		int k = 2;

		// read the values (features) - generate vectors from input data
		List<NamedVector> vectors = getPoints(points);

		// Create input directories for data
		File testData = new File("testdata");

		if (!testData.exists()) {
			testData.mkdir();
		}
		testData = new File("testdata/points");
		if (!testData.exists()) {
			testData.mkdir();
		}

		// Write initial centers
		Configuration conf = new Configuration();

		FileSystem fs = FileSystem.get(conf);

		// Write vectors to input directory
		writePointsToFile(vectors, "testdata/points/file1", fs, conf);

		Path path = new Path("testdata/clusters/part-00000");

		SequenceFile.Writer writer = new SequenceFile.Writer(fs, conf, path,
				Text.class, Kluster.class);

		for (int i = 0; i < k; i++) {
			Vector vec = vectors.get(i);

			// write the initial center here as vec
			Kluster cluster = new Kluster(vec, i,
					new EuclideanDistanceMeasure());
			writer.append(new Text(cluster.getIdentifier()), cluster);
		}

		writer.close();

		// Run K-means algorithm
		KMeansDriver.run(conf, new Path("testdata/points"), new Path(
				"testdata/clusters"), new Path("output"), 0.001, 10, true, 0,
				false);
		SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(
				"output/" + Cluster.CLUSTERED_POINTS_DIR + "/part-m-00000"),
				conf);
		IntWritable key = new IntWritable();

		// Read output values
		WeightedPropertyVectorWritable value = new WeightedPropertyVectorWritable();
		while (reader.next(key, value)) {
			System.out.println(value.toString() + " belongs to cluster "
					+ key.toString());
			
			Map<Text, Text> map = value.getProperties();
			Vector vector = value.getVector();
//			vector.getD
			double weight = value.getWeight();
		}
		reader.close();
	}

}
