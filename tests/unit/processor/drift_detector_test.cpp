/// @file drift_detector_test.cpp
/// @brief Tests for drift detection

#include <gtest/gtest.h>

#include "processor/drift/drift_detector.h"

namespace pyflare::drift {
namespace {

TEST(DriftDetectorTest, FeatureDetectorNoReference) {
    auto detector = DriftDetectorFactory::CreateFeatureDetector();

    std::vector<DataPoint> batch;
    batch.push_back(DataPoint{.id = "1", .features = {1.0, 2.0, 3.0}});

    auto result = detector->Compute(batch);
    EXPECT_FALSE(result.ok());
}

TEST(DriftDetectorTest, FeatureDetectorNoDrift) {
    auto detector = DriftDetectorFactory::CreateFeatureDetector(0.5);

    // Set reference distribution
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({1.0 + i * 0.01, 2.0 + i * 0.01});
    }
    ASSERT_TRUE(detector->SetReference(reference).ok());

    // Test with similar distribution
    std::vector<DataPoint> batch;
    for (int i = 0; i < 50; ++i) {
        batch.push_back(DataPoint{
            .id = std::to_string(i),
            .features = {1.0 + i * 0.01, 2.0 + i * 0.01}
        });
    }

    auto result = detector->Compute(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(result->is_drifted);
    EXPECT_LT(result->score, 0.5);
}

TEST(DriftDetectorTest, FeatureDetectorWithDrift) {
    auto detector = DriftDetectorFactory::CreateFeatureDetector(0.1);

    // Set reference distribution
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        reference.AddSample({1.0, 2.0});
    }
    ASSERT_TRUE(detector->SetReference(reference).ok());

    // Test with very different distribution
    std::vector<DataPoint> batch;
    for (int i = 0; i < 50; ++i) {
        batch.push_back(DataPoint{
            .id = std::to_string(i),
            .features = {100.0, 200.0}  // Very different
        });
    }

    auto result = detector->Compute(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->is_drifted);
    EXPECT_GT(result->score, 0.1);
}

TEST(DriftDetectorTest, EmbeddingDetectorNoDrift) {
    auto detector = DriftDetectorFactory::CreateEmbeddingDetector(0.5);

    // Set reference embeddings
    Distribution reference;
    for (int i = 0; i < 10; ++i) {
        reference.AddSample({1.0, 0.0, 0.0});  // Unit vectors in x direction
    }
    ASSERT_TRUE(detector->SetReference(reference).ok());

    // Test with similar embeddings
    std::vector<DataPoint> batch;
    for (int i = 0; i < 5; ++i) {
        batch.push_back(DataPoint{
            .id = std::to_string(i),
            .embedding = {0.99f, 0.01f, 0.0f}  // Almost the same
        });
    }

    auto result = detector->Compute(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(result->is_drifted);
}

TEST(DriftDetectorTest, DistributionMean) {
    Distribution dist;
    dist.AddSample({1.0, 2.0});
    dist.AddSample({3.0, 4.0});
    dist.AddSample({5.0, 6.0});

    auto mean = dist.Mean();
    ASSERT_EQ(mean.size(), 2);
    EXPECT_DOUBLE_EQ(mean[0], 3.0);
    EXPECT_DOUBLE_EQ(mean[1], 4.0);
}

TEST(DriftDetectorTest, DistributionStdDev) {
    Distribution dist;
    dist.AddSample({1.0});
    dist.AddSample({2.0});
    dist.AddSample({3.0});
    dist.AddSample({4.0});
    dist.AddSample({5.0});

    auto stddev = dist.StdDev();
    ASSERT_EQ(stddev.size(), 1);
    EXPECT_NEAR(stddev[0], 1.5811, 0.001);  // sqrt(2.5)
}

TEST(DriftDetectorTest, DriftTypeToString) {
    EXPECT_EQ(DriftTypeToString(DriftType::kFeature), "feature");
    EXPECT_EQ(DriftTypeToString(DriftType::kEmbedding), "embedding");
    EXPECT_EQ(DriftTypeToString(DriftType::kConcept), "concept");
    EXPECT_EQ(DriftTypeToString(DriftType::kPrediction), "prediction");
}

}  // namespace
}  // namespace pyflare::drift
