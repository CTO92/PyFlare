/// @file mmd_detector_test.cpp
/// @brief Tests for MMD (Maximum Mean Discrepancy) drift detector

#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "processor/drift/mmd_detector.h"

namespace pyflare::drift {
namespace {

class MMDDriftDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.threshold = 0.1;
        config_.num_permutations = 50;  // Fewer for faster tests
        config_.random_seed = 42;  // Reproducible results
        detector_ = std::make_unique<MMDDriftDetector>(config_);
    }

    // Generate random embeddings with mean and variance
    std::vector<std::vector<float>> GenerateEmbeddings(
        size_t count, size_t dim, float mean = 0.0f, float stddev = 1.0f) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(mean, stddev);

        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            std::vector<float> emb(dim);
            for (size_t j = 0; j < dim; ++j) {
                emb[j] = dist(rng);
            }
            embeddings.push_back(std::move(emb));
        }
        return embeddings;
    }

    // Generate unit vectors pointing in a direction
    std::vector<std::vector<float>> GenerateDirectionalEmbeddings(
        size_t count, size_t dim, size_t axis = 0, float noise = 0.01f) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, noise);

        std::vector<std::vector<float>> embeddings;
        for (size_t i = 0; i < count; ++i) {
            std::vector<float> emb(dim, 0.0f);
            emb[axis % dim] = 1.0f;  // Unit vector in axis direction
            for (size_t j = 0; j < dim; ++j) {
                emb[j] += dist(rng);  // Add noise
            }
            // Normalize
            float norm = 0.0f;
            for (float v : emb) norm += v * v;
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (float& v : emb) v /= norm;
            }
            embeddings.push_back(std::move(emb));
        }
        return embeddings;
    }

    MMDConfig config_;
    std::unique_ptr<MMDDriftDetector> detector_;
};

TEST_F(MMDDriftDetectorTest, NoReferenceReturnsError) {
    std::vector<DataPoint> batch;
    batch.push_back(DataPoint{
        .id = "1",
        .embedding = {1.0f, 0.0f, 0.0f}
    });

    auto result = detector_->Compute(batch);
    EXPECT_FALSE(result.ok());
}

TEST_F(MMDDriftDetectorTest, SetReferenceEmbeddings) {
    auto ref_embeddings = GenerateEmbeddings(100, 128);
    auto status = detector_->SetReferenceEmbeddings(ref_embeddings);
    EXPECT_TRUE(status.ok());
}

TEST_F(MMDDriftDetectorTest, NoDriftWithSimilarEmbeddings) {
    // Reference embeddings from same distribution
    auto ref_embeddings = GenerateEmbeddings(100, 64, 0.0f, 1.0f);
    ASSERT_TRUE(detector_->SetReferenceEmbeddings(ref_embeddings).ok());

    // Test with embeddings from same distribution
    auto test_embeddings = GenerateEmbeddings(50, 64, 0.0f, 1.0f);
    auto result = detector_->ComputeFromEmbeddings(test_embeddings);

    ASSERT_TRUE(result.ok());
    // MMD should be relatively low for same distribution
    EXPECT_LT(result->score, 1.0);  // Not perfect but should be relatively low
}

TEST_F(MMDDriftDetectorTest, DetectsDriftWithDifferentDistributions) {
    // Reference: embeddings with mean 0
    auto ref_embeddings = GenerateEmbeddings(100, 64, 0.0f, 1.0f);
    ASSERT_TRUE(detector_->SetReferenceEmbeddings(ref_embeddings).ok());

    // Test: embeddings with very different mean
    auto test_embeddings = GenerateEmbeddings(50, 64, 10.0f, 1.0f);
    auto result = detector_->ComputeFromEmbeddings(test_embeddings);

    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->is_drifted);
    EXPECT_GT(result->score, config_.threshold);
}

TEST_F(MMDDriftDetectorTest, DirectionalDriftDetection) {
    // Reference: embeddings pointing in X direction
    auto ref_embeddings = GenerateDirectionalEmbeddings(100, 128, 0);
    ASSERT_TRUE(detector_->SetReferenceEmbeddings(ref_embeddings).ok());

    // Test: embeddings pointing in Y direction (orthogonal)
    auto test_embeddings = GenerateDirectionalEmbeddings(50, 128, 1);
    auto result = detector_->ComputeFromEmbeddings(test_embeddings);

    ASSERT_TRUE(result.ok());
    // Should detect the directional shift
    EXPECT_TRUE(result->is_drifted || result->score > 0.01);
}

TEST_F(MMDDriftDetectorTest, ComputeViaDataPoints) {
    Distribution reference;
    for (int i = 0; i < 100; ++i) {
        DataPoint point;
        point.embedding = {static_cast<float>(i) / 100.0f, 0.5f, 0.3f};
        reference.AddSample(point.embedding);
    }
    ASSERT_TRUE(detector_->SetReference(reference).ok());

    std::vector<DataPoint> batch;
    for (int i = 0; i < 50; ++i) {
        batch.push_back(DataPoint{
            .id = std::to_string(i),
            .embedding = {static_cast<float>(i) / 50.0f, 0.5f, 0.3f}
        });
    }

    auto result = detector_->Compute(batch);
    ASSERT_TRUE(result.ok());
    EXPECT_GE(result->score, 0.0);
}

TEST_F(MMDDriftDetectorTest, CentroidDrift) {
    auto ref_embeddings = GenerateDirectionalEmbeddings(100, 64, 0);
    ASSERT_TRUE(detector_->SetReferenceEmbeddings(ref_embeddings).ok());

    // Get centroid
    auto centroid = detector_->GetReferenceCentroid();
    EXPECT_EQ(centroid.size(), 64);

    // Centroid should be close to [1, 0, 0, ...]
    EXPECT_GT(centroid[0], 0.9f);

    // Compute centroid drift
    auto diff_embeddings = GenerateDirectionalEmbeddings(50, 64, 32);  // Different axis
    double drift = detector_->ComputeCentroidDrift(diff_embeddings);
    EXPECT_GT(drift, 0.0);  // Should have some drift
}

TEST_F(MMDDriftDetectorTest, GettersAndSetters) {
    EXPECT_EQ(detector_->Name(), "MMDDriftDetector");
    EXPECT_EQ(detector_->Type(), DriftType::kEmbedding);
    EXPECT_DOUBLE_EQ(detector_->GetThreshold(), 0.1);

    detector_->SetThreshold(0.2);
    EXPECT_DOUBLE_EQ(detector_->GetThreshold(), 0.2);

    const auto& config = detector_->GetConfig();
    EXPECT_EQ(config.num_permutations, 50);
}

TEST_F(MMDDriftDetectorTest, SerializeAndLoad) {
    // Set up reference
    auto ref_embeddings = GenerateEmbeddings(50, 32);
    ASSERT_TRUE(detector_->SetReferenceEmbeddings(ref_embeddings).ok());

    // Serialize
    auto state = detector_->SerializeState();
    ASSERT_TRUE(state.ok());
    EXPECT_FALSE(state->empty());

    // Load into new detector
    auto new_detector = std::make_unique<MMDDriftDetector>(config_);
    ASSERT_TRUE(new_detector->LoadState(*state).ok());

    // Verify similar behavior
    auto test_embeddings = GenerateEmbeddings(25, 32);
    auto orig_result = detector_->ComputeFromEmbeddings(test_embeddings);
    auto new_result = new_detector->ComputeFromEmbeddings(test_embeddings);

    ASSERT_TRUE(orig_result.ok());
    ASSERT_TRUE(new_result.ok());
    // Scores should be identical or very close
    EXPECT_NEAR(orig_result->score, new_result->score, 0.01);
}

TEST_F(MMDDriftDetectorTest, EmptyEmbeddingsReturnsError) {
    auto ref_embeddings = GenerateEmbeddings(50, 32);
    ASSERT_TRUE(detector_->SetReferenceEmbeddings(ref_embeddings).ok());

    std::vector<std::vector<float>> empty;
    auto result = detector_->ComputeFromEmbeddings(empty);
    EXPECT_FALSE(result.ok());
}

TEST_F(MMDDriftDetectorTest, DimensionMismatchReturnsError) {
    auto ref_embeddings = GenerateEmbeddings(50, 64);  // 64-dim
    ASSERT_TRUE(detector_->SetReferenceEmbeddings(ref_embeddings).ok());

    auto test_embeddings = GenerateEmbeddings(25, 128);  // 128-dim (mismatch!)
    auto result = detector_->ComputeFromEmbeddings(test_embeddings);
    EXPECT_FALSE(result.ok());
}

TEST_F(MMDDriftDetectorTest, SubsamplingForLargeInputs) {
    MMDConfig config;
    config.max_samples = 20;  // Force subsampling
    config.threshold = 0.1;
    auto detector = std::make_unique<MMDDriftDetector>(config);

    auto ref_embeddings = GenerateEmbeddings(100, 32);
    ASSERT_TRUE(detector->SetReferenceEmbeddings(ref_embeddings).ok());

    auto test_embeddings = GenerateEmbeddings(100, 32);
    auto result = detector->ComputeFromEmbeddings(test_embeddings);
    ASSERT_TRUE(result.ok());
    // Should still produce valid score even with subsampling
    EXPECT_GE(result->score, 0.0);
}

TEST_F(MMDDriftDetectorTest, AutoSigma) {
    // Test median heuristic sigma estimation
    MMDConfig config;
    config.rbf_sigma = 0.0;  // Auto-compute using median heuristic
    config.threshold = 0.1;
    auto detector = std::make_unique<MMDDriftDetector>(config);

    auto ref_embeddings = GenerateEmbeddings(50, 32);
    ASSERT_TRUE(detector->SetReferenceEmbeddings(ref_embeddings).ok());

    auto test_embeddings = GenerateEmbeddings(25, 32);
    auto result = detector->ComputeFromEmbeddings(test_embeddings);
    ASSERT_TRUE(result.ok());
    EXPECT_GE(result->score, 0.0);
}

TEST_F(MMDDriftDetectorTest, FixedSigma) {
    MMDConfig config;
    config.rbf_sigma = 1.0;  // Fixed sigma
    config.threshold = 0.1;
    auto detector = std::make_unique<MMDDriftDetector>(config);

    auto ref_embeddings = GenerateEmbeddings(50, 32);
    ASSERT_TRUE(detector->SetReferenceEmbeddings(ref_embeddings).ok());

    auto test_embeddings = GenerateEmbeddings(25, 32);
    auto result = detector->ComputeFromEmbeddings(test_embeddings);
    ASSERT_TRUE(result.ok());
    EXPECT_GE(result->score, 0.0);
}

// =============================================================================
// Factory Tests
// =============================================================================

TEST(MMDDetectorFactoryTest, CreatesMMDDetector) {
    MMDConfig config;
    config.threshold = 0.15;
    config.num_permutations = 100;

    auto detector = CreateMMDDetector(config);
    ASSERT_NE(detector, nullptr);
    EXPECT_EQ(detector->Name(), "MMDDriftDetector");
    EXPECT_DOUBLE_EQ(detector->GetThreshold(), 0.15);
}

TEST(MMDDetectorFactoryTest, DefaultConfig) {
    auto detector = CreateMMDDetector();
    ASSERT_NE(detector, nullptr);
    EXPECT_EQ(detector->Name(), "MMDDriftDetector");
}

}  // namespace
}  // namespace pyflare::drift
